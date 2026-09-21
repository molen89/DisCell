#!/usr/bin/env python3
"""One-run report: training progress, the disentanglement quadrant, effects.

Reads a finished run directory (TensorBoard events, ``history.jsonl``,
``metrics.json``, ``best.pt``), re-runs one validation sweep from the best
checkpoint, and writes ``report/report.md`` plus every figure it references.
The narrative is fixed; the numbers and figures come from the run, so the
same command reproduces the same report for any run.

Usage::

    python -m discell.model.report --dataset <id> --run reference_best
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths
from discell.model.metrics import principal_curve
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.report")

#: Cells used for the pseudotime cross-checks; matches metrics.MAX_EVAL_CELLS.
SAMPLE = 30_000


# -- pulled from the run directory -----------------------------------------

def extract_tensorboard(run_dir: Path, fig_dir: Path) -> tuple[dict, dict]:
    """Last logged image per figure tag (as PNG files) + full scalar series."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)

    events = sorted(run_dir.glob("events.out.tfevents.*"))
    acc = EventAccumulator(str(events[-1]), size_guidance={"images": 0,
                                                           "scalars": 0})
    acc.Reload()
    images: dict[str, str] = {}
    for tag in acc.Tags()["images"]:
        last = acc.Images(tag)[-1]
        name = tag.split("/")[-1] + ".png"
        (fig_dir / name).write_bytes(last.encoded_image_string)
        images[tag.split("/")[-1]] = name
    scalars = {tag: np.array([(e.step, e.value) for e in acc.Scalars(tag)])
               for tag in acc.Tags()["scalars"]}
    return images, scalars


def progress_figure(scalars: dict, history: list[dict], best_epoch: int,
                    path: Path) -> None:
    """Six panels of what training should look like, best epoch marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.array([h["epoch"] for h in history])
    steps = np.array([h["step"] for h in history])
    # map training steps onto epochs through the eval records
    to_epoch = lambda s: np.interp(s, steps, epochs)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    panels = [
        ("held-out reconstruction",
         [("val/recon", "val recon (per count)", {})], "nats/count"),
        ("identity stays in z",
         [("val/nmi", "z-type NMI", {})], "NMI"),
        ("invariance: probe finds no leak",
         [("val/probe_delta_ce", "probe dCE", {}),
          ("val/probe_noise_floor", "noise floor", {"ls": "--"})], "dCE"),
        ("mirror attractor stays down",
         [("val/mirror_r2", "mirror R2", {}),
          ("val/mirror_r2_permuted", "permuted control", {"ls": "--"})], "R2"),
        ("cycle lands in z, not w",
         [("val/cycle_r2_z_pooled", "z", {}),
          ("val/cycle_r2_w_pooled", "w", {}),
          ("val/cycle_r2_linear_ref_pooled",
           "50-PC linear expression ref", {"ls": "--"})],
         "pooled R2"),
        ("channel usage (KL to prior)",
         [("train/kl_z", "KL_z", {}), ("train/kl_w", "KL_w", {})], "nats"),
    ]
    # old runs logged the linear expression reference as "ceiling"
    alias = ("val/cycle_r2_ceiling_pooled", "val/cycle_r2_linear_ref_pooled")
    if alias[0] in scalars and alias[1] not in scalars:
        scalars[alias[1]] = scalars[alias[0]]
    for ax, (title, series, ylabel) in zip(axes.ravel(), panels):
        for tag, label, style in series:
            if tag not in scalars:
                continue
            xy = scalars[tag]
            ax.plot(to_epoch(xy[:, 0]), xy[:, 1], label=label, lw=1.2, **style)
        ax.axvline(best_epoch, color="0.75", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("training progress (grey line: best checkpoint, joint "
                 "recon + NMI-guard criterion)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- the cross-checks: each channel fails the other's analysis --------------

def cycle_projection_figure(z: np.ndarray, w: np.ndarray, t: np.ndarray,
                            scores: np.ndarray, phase: np.ndarray,
                            groups: list[int], names, path: Path) -> None:
    """z.beta_S vs z.beta_G2M next to the same construction from w.

    The same within-type ridge probe is fitted to both latents; if the cycle
    lives in z only, the z panel shows the G1-blob-plus-arc and the w panel
    shows an unstructured blob at the same construction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase_colour = {0: "#b8b8b8", 1: "#e6772e", 2: "#7d3ac1"}
    fig, axes = plt.subplots(len(groups), 2,
                             figsize=(7.6, 3.4 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, g in enumerate(groups):
        members = np.flatnonzero(t == g)
        target = scores[members] - scores[members].mean(axis=0)
        for c, (latent, label) in enumerate(((z, "z"), (w, "w"))):
            values = latent[members] - latent[members].mean(axis=0)
            gram = values.T @ values + 1e-3 * np.eye(values.shape[1])
            beta = np.linalg.solve(gram, values.T @ target)
            projected = values @ beta
            ax = axes[r, c]
            for p in (0, 1, 2):
                sel = phase[members] == p
                if sel.any():
                    ax.scatter(projected[sel, 0], projected[sel, 1], s=0.9,
                               color=phase_colour[p],
                               label=("G1", "S", "G2M")[p], rasterized=True)
            ax.set_title(f"{label}·β_S vs {label}·β_G2M | {names[g][:22]}",
                         fontsize=8)
            ax.legend(fontsize=5, markerscale=6)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("the same cycle probe on both latents", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pseudotime_cross_check(z: np.ndarray, w: np.ndarray, y: np.ndarray,
                           t: np.ndarray, positions: np.ndarray,
                           edge_i: np.ndarray, edge_j: np.ndarray,
                           rows: np.ndarray, path: Path,
                           seed: int = 0) -> dict:
    """Fit the full-space principal curve on each latent; ask which ordering
    is a *tissue* gradient.

    Two reads per pseudotime, each raw and **type-partialled**: held-out
    ridge R2 from the neighbour composition ``y`` (does the niche predict
    the ordering?), and neighbour coherence -- corr(pt_i, mean of
    graph-neighbour pt) (is the ordering spatially smooth?). The partialled
    variant residualises pseudotime and ``y`` against per-type means first
    (the mirror_r2 pattern): z legitimately carries type, and type is
    spatially predictable through homophily, so the raw numbers partly read
    type -- the beyond-type numbers are the honest contrast.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    sub = rows if len(rows) <= SAMPLE else np.sort(
        rng.choice(rows, SAMPLE, replace=False))
    local = {node: k for k, node in enumerate(sub)}

    # graph edges with both endpoints in the sample, as directed pairs
    keep = np.isin(edge_i, sub) & np.isin(edge_j, sub)
    src = np.array([local[v] for v in np.concatenate(
        [edge_i[keep], edge_j[keep]])])
    dst = np.array([local[v] for v in np.concatenate(
        [edge_j[keep], edge_i[keep]])])

    half = rng.permutation(len(sub))
    fit_rows, test_rows = half[: len(sub) // 2], half[len(sub) // 2:]

    def niche_r2(target: np.ndarray, design_block: np.ndarray) -> float:
        design = np.hstack([design_block, np.ones((len(sub), 1))])
        gram = design[fit_rows].T @ design[fit_rows] \
            + 1e-3 * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design[fit_rows].T @ target[fit_rows])
        residual = target[test_rows] - design[test_rows] @ coef
        return float(1.0 - residual.var() / target[test_rows].var())

    def coherence(target: np.ndarray) -> float:
        total = np.zeros(len(sub)); count = np.zeros(len(sub))
        np.add.at(total, dst, target[src]); np.add.at(count, dst, 1.0)
        touched = count > 0
        return float(np.corrcoef(target[touched],
                                 total[touched] / count[touched])[0, 1])

    t_sub = t[sub]
    y_centred = y[sub].copy()
    for g in np.unique(t_sub):
        y_centred[t_sub == g] -= y_centred[t_sub == g].mean(axis=0)

    out: dict[str, dict] = {}
    order = {}
    for label, latent in (("z", z), ("w", w)):
        pt, _ = principal_curve(latent[np.searchsorted(rows, sub)])
        order[label] = pt
        pt_centred = pt.copy()
        for g in np.unique(t_sub):
            pt_centred[t_sub == g] -= pt_centred[t_sub == g].mean()
        out[label] = {
            "niche_r2": niche_r2(pt_centred, y_centred),
            "neighbour_coherence": coherence(pt_centred),
            "niche_r2_raw": niche_r2(pt, y[sub]),
            "neighbour_coherence_raw": coherence(pt),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, label in zip(axes, ("w", "z")):
        ax.scatter(positions[sub, 0], positions[sub, 1], c=order[label],
                   cmap="viridis", s=0.8, rasterized=True)
        ax.set_title(f"tissue painted by {label}-pseudotime (beyond-type "
                     f"niche R2 {out[label]['niche_r2']:.2f}, "
                     f"coherence {out[label]['neighbour_coherence']:.2f})",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


# -- assembly ---------------------------------------------------------------

def build(args: argparse.Namespace) -> Path:
    import torch

    from discell.model.networks import DisCell

    run_dir = paths.dataset(args.dataset).root / "runs" / args.run
    out_dir = run_dir / "report"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = [json.loads(line) for line in
               (run_dir / "history.jsonl").read_text().splitlines()]
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    payload["config"].setdefault("gat_sources", "type_z")   # pre-field era
    payload["config"].setdefault("subtract_leak", False)
    payload["config"].setdefault("gat_sink", False)
    config = TrainConfig(**payload["config"])
    run_meta = json.loads((run_dir / "config.json").read_text())

    images, scalars = extract_tensorboard(run_dir, fig_dir)
    progress_figure(scalars, history, metrics["best"]["epoch"],
                    fig_dir / "progress.png")

    # -- one validation sweep from the best checkpoint ---------------------
    data = assemble(args.dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = DisCell(n_genes=data.x.shape[1], n_types=len(data.p_t),
                    phi_dim=data.phi.shape[1],
                    median_counts=data.median_counts, d_z=config.d_z,
                    d_w=config.d_w, hidden=config.hidden,
                    gat_dim=config.gat_dim, heads=config.heads,
                    gat_sources=config.gat_sources,
                    subtract_leak=config.subtract_leak,
                    gat_sink=config.gat_sink).to(device)
    model.load_state_dict(payload["model"])
    trainer = Trainer(config, data)
    trainer.model = model.eval()
    sweep = trainer._sweep(trainer.val_batches)
    rows = sweep["nodes"]
    order = np.argsort(rows)
    rows, mu_z, mu_w = rows[order], sweep["mu_z"][order], sweep["mu_w"][order]

    cyc = data.cycle
    names = [str(n) for n in data.type_names]
    groups = [g for g in cyc["cycling_types"]
              if "nassigned" not in names[g]][:2]
    scores = np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1)
    cycle_projection_figure(mu_z, mu_w, data.t[rows], scores[rows],
                            cyc["phase"][rows], list(groups), names,
                            fig_dir / "cycle_projection_zw.png")
    pseudo = pseudotime_cross_check(
        mu_z, mu_w, data.graph.y, data.t, data.positions, data.graph.edge_i,
        data.graph.edge_j, rows, fig_dir / "pseudotime_tissue.png",
        seed=config.seed)

    text = render(args.run, run_meta, config, metrics, history, images,
                  pseudo, names, groups)
    text += (atlas_section(run_dir) + transport_section(run_dir)
             + lr_map_section(run_dir) + a4_section(run_dir))
    (out_dir / "report.md").write_text(text)
    log.info("wrote %s", out_dir / "report.md")
    return out_dir


def atlas_section(run_dir: Path) -> str:
    """The section-6 w-programme atlas, when it has been run for this model."""
    path = run_dir / "atlas" / "atlas.json"
    if not path.exists():
        return ""
    atlas = json.loads(path.read_text())
    programmes = atlas["programs"]
    r, d_w = atlas["rank"], atlas["d_w"]
    spectrum = ", ".join(f"{f:.2f}" for f in atlas["variance_fraction"])
    cross = atlas.get("cross_seed", {})
    block_names = atlas.get("driver_blocks",
                            ["composition_y", "phi_pcs", "landmark_distances"])
    head = ("| # | share of w var | share of shift | hallmark (rank test, BH "
            "q ≤ 0.05) | Moran I (null 97.5%) | driver R² joint | partial: "
            + " / ".join(n.replace("_", " ") for n in block_names)
            + " | most modulated type |")
    sep = "|---|---|---|---|---|---|---|---|"
    if cross:
        head += " cross-seed \\|cos\\| vs " + " / ".join(cross) + " |"
        sep += "---|"
    rows, figures = [head, sep], []
    for p in programmes:
        top_hm = next((h for h in p["hallmarks"] if h.get("significant")), None)
        hallmark = (f"{top_hm['hallmark'][:28]} (q={top_hm['q']:.1e}, "
                    f"AUC {top_hm['auc']:.2f})" if top_hm else "unlabelled")
        top_type = next(iter(p["type_activity"]), "-")
        drv = p["drivers"]
        partial = " / ".join(f"{drv['partial'][n]:.2f}" for n in block_names)
        row = (f"| {p['program']} | {p['variance_share']:.2f} | "
               f"{p['shift_share']:.2f} | {hallmark} | {p['moran_I']:.2f} "
               f"({p['moran_null_hi']:.2f}) | {drv['joint']:.2f} | {partial} | "
               f"{top_type[:26]} |")
        if cross:
            matched = [c["axis_cosine"][p["program"]]
                       if p["program"] < len(c["axis_cosine"]) else None
                       for c in cross.values()]
            row += " " + " / ".join(f"{m['cosine']:.2f}" if m else "–"
                                    for m in matched) + " |"
        rows.append(row)
        figures.append(f"![programme {p['program']}]"
                       f"(../atlas/program_{p['program']}.png)")
    cross_text = "".join(
        f" Against `{other}` (rank {c['rank']}): shift-space overlap "
        f"{c['shift_overlap']['this_inside_other']:.2f} of this model's "
        f"realised-shift variance lies inside that model's programme span "
        f"({c['shift_overlap']['other_inside_this']:.2f} the other way); "
        f"per-axis \\|cos\\| in the table."
        for other, c in cross.items()) or (
        " No other seed's atlas was passed (`--compare-runs`), so axis "
        "reproducibility is not judged here.")
    rec = atlas.get("label_recurrence", {})
    if rec.get("n_runs", 0) > 1:
        sets = "; ".join(f"`{run}` {', '.join(names) or '(unlabelled)'}"
                         for run, names in rec["label_sets"].items())
        recurring = ", ".join(rec["recurring"]) or "none"
        recurrence = (
            f"**Label recurrence** over {rec['n_runs']} fits — {sets}. "
            f"Labels significant in ≥ {rec['minimum']} fits: **{recurring}**. "
            "A label that does not recur is not a name for the programme; "
            "the programme is reported unlabelled.")
    else:
        recurrence = ("**Label recurrence** was not read here: no other fit's "
                      "atlas was passed (`--compare-runs` for seeds, "
                      "`--compare-atlas` for other slides). A single fit's "
                      "label is a hypothesis, not a name.")
    landmarks = atlas.get("landmark_classes", [])
    landmark_note = (
        f"Landmark classes on this slide: {', '.join(landmarks)}."
        if landmarks else
        "This slide has **no landmark classes** — the inventory is built from "
        "annotated type names and these labels are unsupervised clusters, so "
        "the landmark driver block is dropped rather than fed an empty "
        "inventory. Its absence is a missing question, not a zero driver.")
    return f"""

## The w-programme atlas (doc-08 section 6)

**What a programme is.** w is the cell's context response: the decoder adds
B·w to the intrinsic profile a(z), so a *direction* v in w-space is a gene
programme with loadings B·v -- the realised expression shift the model
attributes to context along that direction. A programme is a direction and
its loadings; it is not a coordinate of w, and it is not a cluster of cells.
Two things about w carry no information and are removed before reading: raw
w dimensions are rotation-arbitrary, and w's per-type mean is a gauge --
(a(z) − B·μ_t, w + μ_t) is the same model for any per-type constant μ_t
(issues V12) -- so raw ‖w‖ and per-type ‖w‖ rankings mean nothing and are
withdrawn; only the within-type variation of w, read through a fixed
procedure, reproduces.

**How.** (1) w is gauge-centred within type at the reference context, so
every read below is of Δ_i = B·[w_i − w̄_t], the shift relative to the
average context of the cell's own type. (2) The programmes are the r
principal directions of cov(w) carrying ≥ {100 * atlas['rank_var_fraction']:.0f}%
of its trace -- the **effective rank** (issues V10) -- varimax-rotated
*within* that r-dimensional subspace in gene space so each programme has a
sparse signature; the rotation is applied to coordinates and loadings
together, so the decoder's output is unchanged. Activity is judged by that
spectrum, never by the variance of a rotated coordinate: a rank-2 w rotated
onto 6 axes gives six collinear coordinates that would all pass. The
remaining d_w − r directions carry no variance: they are **null directions**,
not spare programmes -- the data do not constrain their B columns, which is
why matched-column B correlation across seeds reads as instability and is
not reported (issues V11). (3) Per programme: its share of w's variance and
of the realised shift's variance; its signature (top ± loadings) with a
**rank-based** hallmark label -- a Mann-Whitney test on the ranks of the
*full* loading vector, set members against the rest of the expressed panel,
so no arbitrary top-k cut enters and the background is the panel actually
measured; sets with < 5 panel genes are untestable and dropped, BH is
applied per programme, and a label ships only at q ≤ 0.05. Hallmarks
*annotate* programmes; no pathway is defined from a programme or tested on
the data it was fitted to. (4) Moran's I of the coordinate on the neighbour
graph against a permutation null (territoriality). (5) Held-out
spatial-block ridge R² of the coordinate from the context blocks --
neighbour composition, image PCs, and landmark distances where the slide has
landmarks -- with target and blocks both within-type-centred (type-
partialled: the drivers are the cell's context, not its type identity via
homophilous composition), reported joint and partial (the blocks overlap:
vessel density itself varies rim to core). (6) The within-type variance of
the coordinate per type -- which types the programme modulates, offset-free
by construction. (7) Three concordance reads: per-axis cross-seed \\|cos\\| and
shift-space overlap on the same slide; hallmark-label recurrence across
seeds and across slides (label *sets*, since programme order and gene panels
need not agree); and κ-survival, which is sweep-internal.

**Evaluated by.** r against d_w and the eigen-spectrum; each programme's
variance share; the BH-gated rank-test label with its q and AUC; Moran's I
against its null; joint and partial driver R²; cross-seed per-axis \\|cos\\|
and shift-space overlap; and whether the label recurs in ≥ 2 fits.

**What is wished for.** Few programmes (r ≪ d_w), each territorial (Moran's
I well above the null), context-explained (high joint driver R² with
nameable partial shares -- a composition-level effect, not a communication
claim), carrying a hallmark label that **replicates across seeds and
slides**, and reproducing as a subspace (per-axis \\|cos\\| ≳ 0.8, shift-space
overlap ≳ 0.95). A programme with a low cross-seed cosine is seed-specific
structure of the optimum, not tissue. A programme whose label does not
recur is reported as **unlabelled** -- never named from one fit.

**This model.** r = {r} of d_w = {d_w} directions carry variance
(eigen-fractions of within-type cov(w): {spectrum}); {d_w - r} null
directions.{cross_text}

{recurrence}

{landmark_note}

{chr(10).join(rows)}

κ-survival: {atlas.get('kappa_survival', 'see experiments/')}

""" + "\n\n".join(figures) + "\n"


TRANSPORT_ROWS = (("counterfactual", "counterfactual (program + leak, z fixed)"),
                  ("counterfactual_phi_fixed", "same, Phi frozen at the type mean"),
                  ("program_only", "program channel only"),
                  ("leak_only", "leak channel only"),
                  ("full", "model account (selection also free)"))


def transport_table(summaries: dict) -> str:
    """One table a reader can follow without the code: tiers down the
    columns, the predictors down the rows, plus the bars each tier is
    judged on. ``summaries`` maps a column label to a tier summary."""
    cols = list(summaries)
    head = "| predictor (mean held-out R^2) | " + " | ".join(cols) + " |\n"
    head += "|---" * (len(cols) + 1) + "|\n"
    body = ""
    for key, label in TRANSPORT_ROWS:
        body += f"| {label} | " + " | ".join(
            f"{summaries[c][key]:.3f}" for c in cols) + " |\n"
    extras = (("n_panels", "panels", "{:.0f}"),
              ("n_trusted", "of which trusted (ceiling >= 0.5)", "{:.0f}"),
              ("noise_ceiling", "noise ceiling of the observed shift", "{:.3f}"),
              ("counterfactual_of_ceiling", "**share of the ceiling taken"
               " (the headline)**", "{:.2f}"),
              ("top_gene_overlap", "top-50 predicted-up genes in the observed"
               " top 50 (fraction)", "{:.2f}"),
              ("top_gene_chance", "... chance level for that overlap (50/G)",
               "{:.2f}"),
              ("median_slope", "median calibration slope", "{:.2f}"),
              ("interventionable_share", "interventionable share (Phi-fixed"
               " / real-Phi)", "{:.2f}"),
              ("selection_share", "selection share (model account -"
               " counterfactual)", "{:.3f}"))
    for key, label, fmt in extras:
        body += f"| *{label}* | " + " | ".join(
            fmt.format(summaries[c].get(key, float("nan"))) for c in cols
            ) + " |\n"
    body += "| *counterfactual beats both channels* | " + " | ".join(
        f"{summaries[c]['full_beats_both']}/{summaries[c]['n_panels']}"
        for c in cols) + " |\n"
    return head + body


def transport_distribution_block(run_dir: Path) -> str:
    """The distribution-level transport read, as its own sub-block.

    The mean read above scores a difference of niche means. This one moves
    a population and asks whether it lands on the population that was
    already there. Absent until the read has been run."""
    path = run_dir / "transport" / "transport_distribution.json"
    if not path.exists():
        return ""
    res = json.loads(path.read_text())
    cols = [("pairwise A→B, as pre-reg.", ("summary", "pairwise"),
             ("pooling", "agreement_with_mean_read")),
            ("leave-one-out, as pre-reg.", ("summary", "leave_one_out"),
             None),
            ("pairwise A→B, count-matched",
             ("summary_count_matched", "pairwise"),
             ("pooling_count_matched",
              "agreement_with_mean_read_count_matched")),
            ("leave-one-out, count-matched",
             ("summary_count_matched", "leave_one_out"), None)]
    summaries = [res.get(a, {}).get(b, {}) for _, (a, b), _ in cols]
    if not summaries[0].get("n_panels"):
        return ""
    head = "| | " + " | ".join(c[0] for c in cols) + " |\n"
    head += "|---" * (len(cols) + 1) + "|\n"
    body = ""
    for key, label, fmt in (
            ("n_panels", "panels", "{:.0f}"),
            ("median_gap_closed", "**median gap closed**", "{:.2f}"),
            ("q25_gap_closed", "gap closed, 25th pct", "{:.2f}"),
            ("q75_gap_closed", "gap closed, 75th pct", "{:.2f}"),
            ("median_mmd2_transported", "median MMD² transported", "{:.4g}"),
            ("median_mmd2_untransported", "median MMD² untransported",
             "{:.4g}"),
            ("median_mmd2_floor", "median MMD² floor (sampling noise)",
             "{:.4g}"),
            ("median_mmd2_observed_source",
             "median MMD² observed source (no model)", "{:.4g}"),
            ("median_gap_closed_type_mean",
             "gap closed by the type-mean predictor", "{:.2f}"),
            ("median_energy_transported",
             "median energy distance transported", "{:.4g}"),
            ("median_energy_untransported",
             "median energy distance untransported", "{:.4g}"),
            ("median_ci_width", "median paired-bootstrap CI width",
             "{:.4g}")):
        body += f"| {label} | " + " | ".join(
            fmt.format(sm.get(key, float("nan"))) for sm in summaries) + " |\n"
    for key, label in (("n_improved", "*transported below untransported*"),
                       ("n_ci_excludes_zero",
                        "*... with the bootstrap CI excluding 0*"),
                       ("n_beats_type_mean",
                        "*transported below the type-mean predictor*")):
        body += f"| {label} | " + " | ".join(
            f"{sm.get(key, '?')}/{sm.get('n_panels', '?')}"
            for sm in summaries) + " |\n"
    table = head + body

    lines = []
    for label, _, extra in cols:
        if extra is None:
            continue
        pool = res.get(extra[0], {})
        agree = res.get(extra[1], {})
        agree_txt = (f"Spearman {agree['spearman']:.2f} (n={agree['n']}, "
                     f"p={agree['p_value']:.1e})"
                     if agree.get("available") else "not computed")
        lines.append(
            f"- **{label.split(',')[1].strip()}**: pooling gives the tighter "
            f"CI in {pool.get('n_loo_tighter', '?')} of "
            f"{pool.get('n_compared', '?')} shared panels (median width "
            f"{pool.get('median_loo_ci_width', float('nan')):.3g} pooled vs "
            f"{pool.get('median_pairwise_ci_width', float('nan')):.3g} "
            f"pairwise); the pooled source's median gap closed is "
            f"{pool.get('median_gap_drop', float('nan')):+.2f} relative to "
            f"the pairwise one (positive = pooling costs that much). "
            f"Agreement with the mean read (gap closed vs counterfactual "
            f"R²): {agree_txt}.")
    notes = "\n".join(lines)
    cm, cl = summaries[2], summaries[3]
    pool_cm = res.get("pooling_count_matched", {})
    agree_cm = res.get("agreement_with_mean_read_count_matched", {})

    def verdict(ok):
        return "**met**" if ok else "**not met**"
    n_pw = max(cm.get("n_panels", 0), 1)
    bars = (
        f"**The bars on this run (count-matched columns).** (1) transported "
        f"below untransported in {cm.get('n_improved', 0)}/{cm.get('n_panels', 0)}"
        f" panels, CI excluding zero in {cm.get('n_ci_excludes_zero', 0)} — "
        f"{verdict(cm.get('n_improved', 0) > n_pw / 2)}. (2) median gap "
        f"closed {cm.get('median_gap_closed', float('nan')):.2f} pairwise / "
        f"{cl.get('median_gap_closed', float('nan')):.2f} pooled; pooling "
        f"tighter in {pool_cm.get('n_loo_tighter', 0)}/"
        f"{pool_cm.get('n_compared', 0)} and costing "
        f"{pool_cm.get('median_gap_drop', float('nan')):.2f} of gap closed — "
        f"{verdict(pool_cm.get('n_loo_tighter', 0) > pool_cm.get('n_compared', 1) / 2 and pool_cm.get('median_gap_drop', 1.0) <= 0.1)}"
        f". (3) transported below the type-mean predictor in "
        f"{cm.get('n_beats_type_mean', 0)}/{cm.get('n_panels', 0)} — "
        f"{verdict(cm.get('n_beats_type_mean', 0) > n_pw / 2)}. (4) Spearman "
        f"with the mean read "
        f"{agree_cm.get('spearman', float('nan')):.2f} — "
        f"{verdict(agree_cm.get('spearman', 0.0) >= 0.5)}.")
    return f"""

### The same check at the distribution level (MMD read)

**How.** The mean read above scores one number per gene: the difference of
two niche *means*. But transport is a population being moved, so this
sub-block moves it. For a type *t* and a target niche A: take the held-out
cells of *t* that are somewhere else (niche B for the **pairwise** version,
*every* other niche for the **leave-one-niche-out** version), keep each cell
at **its own z**, give it A's context (the prior head at A's mean context)
and A's leak source (κ times A's mean foreign influx), and decode. That
gives a cloud of predicted compositions. Compare it with the cloud of cells
that actually live in A — their raw depth-normalised counts — as
*distributions*: MMD² (unbiased) with a Gaussian kernel on the square-root
(Hellinger) map, bandwidth = the median pairwise distance inside the target,
energy distance reported beside it. Four references on the same cells: the
**floor** (two random halves of the target: pure sampling noise, the best
any predictor can do), the **untransported** cloud (the same source cells
decoded at their *own* contexts: what transport has to remove), the
**type-mean** predictor (one point, the target's mean composition,
replicated) and the **observed source** (the raw niche difference, no model
at all). Both sides are subsampled to the same size (at most 2,000 cells),
model quantities come from training tiles, every cell scored comes from
held-out tiles, and the CI is a 200-draw paired bootstrap over cells. The
headline is **gap closed** = (untransported − transported) / (untransported
− floor): the fraction of the distance to the target population that the
transport removes. 1 = indistinguishable from the cells that were there;
0 = no better than not transporting; negative = transport made it worse.

**One addition to the construction, reported beside it and never instead.**
As pre-registered, a cloud of *predicted rates* is compared with a cloud of
*raw multinomial compositions*. At Xenium depth those are not the same kind
of object: the target's own shot noise is far larger than any difference
between two rate clouds, so it enters every distance as an almost constant
offset and compresses gap closed towards zero for a reason that has nothing
to do with transport. The **count-matched** columns therefore draw counts
from each predicted rate vector at a depth drawn from the target cells, so
both sides carry the same sampling geometry. The floor and the references
are unchanged. Read the count-matched columns as the answer and the
as-pre-registered columns as the record.

**Evaluated by.** (1) transported MMD² below untransported on a majority of
panels, with the paired bootstrap CI excluding zero; (2) gap closed reported
as a distribution, and the pooled leave-one-out version must have a
*tighter* CI than the pairwise one on the same target and a median gap
closed no more than 0.1 below it — if pooling hurts, z is not context-free
in the way the pooled read assumes, and that is the finding; (3) the
transported cloud must not collapse onto the type mean: transported MMD²
below the type-mean predictor's on a majority, else the read is only the
mean shift again; (4) the two instruments must call the same panels good:
Spearman between gap closed and the mean read's counterfactual R² of at
least 0.5.

{table}

{notes}

{bars}

**What can be concluded.** A median gap closed of
{cm.get('median_gap_closed', float('nan')):.2f} (pairwise, count-matched)
says the model's account of a neighbourhood moves a population that fraction
of the way onto the population that actually lives there, with the cells'
own intrinsic states untouched — a statement about *distributions*, not just
means, and one a reader can take without R². The leave-one-out version is
the intrinsic-z test: cells gathered from every other context are asked to
land on one target, and pooling is supposed to help, not hurt. **Not
concluded**: nothing per cell (the clouds are matched as populations, never
cell to cell); nothing about panels whose target has too few held-out cells
to match sizes (reported as insufficient and dropped); nothing from the
as-pre-registered columns about *sizes* of effects, since their scale is set
by shot noise; and no claim that a gap closed near 1 means the model is
right about *why* — MMD² is blind to which genes moved, which is what the
mean read and the top-50 gene overlap are for.

![transport distribution read](../transport/transport_distribution.png)
{transport_second_round_block(run_dir)}
"""


def transport_second_round_block(run_dir: Path) -> str:
    """Model-vs-model MMD and the matched-twin read (devlog 2026-09-21,
    second round), with their HVG companions. Absent until run."""
    dist_path = run_dir / "transport" / "transport_distribution.json"
    twin_path = run_dir / "transport" / "transport_twins.json"
    if not dist_path.exists():
        return ""
    res = json.loads(dist_path.read_text())
    cols = [(label, res.get(block, {}).get(tier, {}))
            for label, block, tier in (
                ("pairwise, all genes", "summary_model", "pairwise"),
                ("leave-one-out, all genes", "summary_model",
                 "leave_one_out"),
                ("pairwise, HVG 1000", "summary_model_hvg", "pairwise"),
                ("leave-one-out, HVG 1000", "summary_model_hvg",
                 "leave_one_out"))]
    cols = [c for c in cols if c[1].get("n_panels")]
    if not cols:
        return ""
    head = "| | " + " | ".join(c[0] for c in cols) + " |\n"
    head += "|---" * (len(cols) + 1) + "|\n"
    body = ""
    for key, label, fmt in (
            ("n_panels", "panels", "{:.0f}"),
            ("median_gap_closed", "**median gap closed**", "{:.2f}"),
            ("q25_gap_closed", "gap closed, 25th pct", "{:.2f}"),
            ("q75_gap_closed", "gap closed, 75th pct", "{:.2f}"),
            ("median_gap_closed_type_mean",
             "**gap closed by the type-mean predictor**", "{:.2f}"),
            ("median_mmd2_transported", "median MMD² transported", "{:.4g}"),
            ("median_mmd2_untransported", "median MMD² untransported",
             "{:.4g}"),
            ("median_mmd2_floor", "median MMD² floor", "{:.4g}")):
        body += f"| {label} | " + " | ".join(
            fmt.format(sm.get(key, float("nan"))) for _, sm in cols) + " |\n"
    for key, label in (("n_improved", "*transported below untransported*"),
                       ("n_ci_excludes_zero", "*... CI excluding 0*"),
                       ("n_beats_type_mean",
                        "*transported below the type-mean predictor*")):
        body += f"| {label} | " + " | ".join(
            f"{sm.get(key, '?')}/{sm.get('n_panels', '?')}"
            for _, sm in cols) + " |\n"
    model_table = head + body

    twin_table = ""
    if twin_path.exists():
        tw = json.loads(twin_path.read_text())
        tcols = [(label, tw.get(block, {}).get(tier, {}))
                 for label, block, tier in (
                     ("pairwise, all genes", "summary", "pairwise"),
                     ("leave-one-out, all genes", "summary", "leave_one_out"),
                     ("pairwise, HVG 1000", "summary_hvg", "pairwise"),
                     ("leave-one-out, HVG 1000", "summary_hvg",
                      "leave_one_out"))]
        tcols = [c for c in tcols if c[1].get("n_panels")]
        if tcols:
            h = "| | " + " | ".join(c[0] for c in tcols) + " |\n"
            h += "|---" * (len(tcols) + 1) + "|\n"
            b = ""
            for key, label, fmt in (
                    ("n_panels", "panels", "{:.0f}"),
                    ("median_gap_closed", "**median gap closed**", "{:.2f}"),
                    ("frac_gap_closed_positive",
                     "fraction of panels with gap closed > 0", "{:.2f}"),
                    ("median_twin_margin", "**median twin margin**",
                     "{:.3f}"),
                    ("frac_twin_margin_positive",
                     "fraction of panels with twin margin > 0", "{:.2f}"),
                    ("median_distance_transported",
                     "median Hellinger, transported twin", "{:.4f}"),
                    ("median_distance_untransported",
                     "median Hellinger, untransported twin", "{:.4f}"),
                    ("median_distance_random",
                     "median Hellinger, random source cell", "{:.4f}"),
                    ("median_distance_floor",
                     "median Hellinger, floor (two target cells)",
                     "{:.4f}")):
                b += f"| {label} | " + " | ".join(
                    fmt.format(sm.get(key, float("nan")))
                    for _, sm in tcols) + " |\n"
            for key, label in (
                    ("n_ci_vs_untransported_excludes_zero",
                     "*CI on (transported − untransported) excludes 0*"),
                    ("n_ci_vs_random_excludes_zero",
                     "*CI on (transported − random) excludes 0*")):
                b += f"| {label} | " + " | ".join(
                    f"{sm.get(key, '?')}/{sm.get('n_panels', '?')}"
                    for _, sm in tcols) + " |\n"
            twin_table = f"""

#### Read B — matched twins (per cell)

For every target cell, its nearest source cell in μ_z (same type, from the
source niche(s)) is transported into the target niche and compared *cell to
cell* with the target cell's own decoded vector, by Hellinger distance.
References per cell: the same twin **untransported**, a **random** same-type
source cell transported (does matching on z buy anything), and the **floor**,
the target cell's z-nearest other target cell, both decoded in the target
niche. Gap closed = (untransported − transported)/(untransported − floor);
twin margin = (random − transported)/random. CIs are paired bootstraps over
cells.

{h + b}

![matched twins](../transport/transport_twins.png)
"""

    return f"""

#### Read A — model-vs-model MMD

Same panels and the same Hellinger-map MMD², but the target cloud is now the
target cells' **own decoded probability vectors** (their posterior-mean z and
w, their own context and leak) instead of their raw counts. Both sides are
then smooth model outputs, so reconstruction error and shot noise leave the
comparison and no count-matched companion is needed; the floor is two halves
of the decoded target and the type-mean predictor is the decoded target's
mean replicated. The **HVG** columns restrict every probability vector to the
top-1000 Scanpy `seurat` highly-variable genes of the training cells and
renormalise. The question this read exists for is the type-mean row: if the
type-mean predictor still closes ≈1.0, the within-niche spread of decoded
cells is itself tiny and the spread limit is in the model, not in shot noise.

{model_table}
{twin_table}"""



def transport_section(run_dir: Path) -> str:
    """The section-7 counterfactual transport check, when present."""
    path = run_dir / "transport" / "transport.json"
    if not path.exists():
        return ""
    res = json.loads(path.read_text())
    band_path = run_dir / "transport" / "transport_tumour-band.json"
    band = json.loads(band_path.read_text()) if band_path.exists() else None
    ex = res.get("summary", {}).get("extrapolation", {})
    sup = res.get("summary", {}).get("supported", {})
    trusted = res.get("summary", {}).get("extrapolation_trusted", {})
    if not ex:
        return ""
    kappa = res.get("kappa", "?")
    summaries = {"composition, extrapolation": ex}
    if sup:
        summaries["composition, supported"] = sup
    if band:
        for tier, label in (("supported", "annotation, supported"),
                            ("supported_trusted",
                             "annotation, supported + trusted")):
            if band.get("summary", {}).get(tier):
                summaries[label] = band["summary"][tier]
    table = transport_table(summaries)
    ceiling = ex.get("noise_ceiling", float("nan"))
    conclusion = (
        f"On held-out tissue the model forecasts the per-gene shift a cell type "
        f"undergoes between two neighbourhoods, and both channels are needed: "
        f"the two-channel counterfactual beats the response channel alone and the "
        f"leak channel alone in {ex['full_beats_both']} of {ex['n_panels']} panels, "
        f"so neither 'it is all biology' nor 'it is all spillage' describes this "
        f"slide. **The headline is the fraction of the noise ceiling the "
        f"counterfactual takes: {ex.get('counterfactual_of_ceiling', float('nan')):.2f} "
        f"of what is reachable** — the raw R² ({ex['counterfactual']:.3f}) belongs "
        f"beside it and never alone, because it is bounded by the "
        f"data, not the model: the observed shifts have a mean noise ceiling of "
        f"{ceiling:.3f}, i.e. most panels contain almost nothing measurable. "
        f"Without any R² at all: of the 50 genes the counterfactual predicts to "
        f"rise most, {50 * ex.get('top_gene_overlap', float('nan')):.0f} on average "
        f"are in the observed top 50, against a chance level of "
        f"{50 * ex.get('top_gene_chance', float('nan')):.1f}. On the "
        f"{ex.get('n_trusted', 0)} panels whose observation is reliable the "
        f"counterfactual reads "
        f"{trusted.get('counterfactual', float('nan')):.3f} at slope "
        f"{trusted.get('median_slope', float('nan')):.2f}, beating both channels in "
        f"{trusted.get('full_beats_both', '?')} of {trusted.get('n_panels', '?')} — "
        f"quote the two numbers together, never the first alone. Freezing the image "
        f"context leaves the forecast unchanged (interventionable share "
        f"{ex.get('interventionable_share', float('nan')):.2f}), so the transported "
        f"effect answers to neighbour composition, the part of a niche an "
        f"intervention could set. The selection share "
        f"({ex.get('selection_share', float('nan')):.3f}) is the part of an observed "
        f"niche difference that is which cells live there rather than what the "
        f"neighbourhood does to them. Not concluded: any per-cell counterfactual "
        f"(out of scope by design), any statement at κ outside the range where the "
        f"slope stays in [0.8, 1.2] (see the κ table), and anything about panels "
        f"below the trust threshold.")
    # the standalone deliverable: one table a reader can follow without
    # the code or the report around it
    (run_dir / "transport" / "transport_table.md").write_text(
        f"# Counterfactual transport, run {res.get('run', '?')} "
        f"(kappa {kappa})\n\n" + table)
    band_n = (band.get("summary", {}).get("supported", {}).get("n_panels", 0)
              if band else "not run")
    flagged = [p for p in res["panels"] if p.get("overlap_flag")]
    top = max(flagged, key=lambda p: p["counterfactual"]["r2"]) \
        if flagged else None
    example = ""
    if top is not None:
        example = f"""**A worked panel** — the best-predicted one:
{top['type']}, niches {top['pair'][0]} vs {top['pair'][1]}. The
counterfactual (context response + new neighbours' influx, z fixed)
predicts the per-gene shift with held-out R²
**{top['counterfactual']['r2']:.2f}** at slope
{top['counterfactual']['slope']:.2f}; the program channel alone reaches
{top['program_only']['r2']:.2f} and the leak channel alone
{top['leak_only']['r2']:.2f} — a substantial part of this type's
apparent between-niche signature is its neighbours' transcripts,
quantified gene by gene.

"""
    return f"""

## Counterfactual transport check (doc-08 section 7)

**The question.** If w really captures how a neighbourhood changes a
cell's expression, the model must be able to forecast: "cells of type t
in niche B vs the same type in niche A — how does each gene's measurement
differ?" — and the forecast must match reality on tissue the model never
trained on. This is the counterfactual claim at the only level it is
answerable: the same *type* across contexts, averaged (per-cell
counterfactuals are out of scope by design).

**How, step by step:**

1. **Neighbourhood kinds**: k-means on neighbour composition defines the
   niches — data-defined labels for kinds of surroundings, no latents
   involved.
2. **A panel** = one type × one niche pair, kept only when the type has
   enough cells on both sides (training and held-out).
3. **Spatial split**: every model quantity comes from training tiles;
   every observation from held-out tiles.
4. **The predicted per-gene shift**, two channels: the *program channel*
   pushes the difference of mean context priors through B (the expression
   response w attributes to swapping the neighbourhood); the *leak
   channel* is the change in foreign influx — kappa times the difference
   of mean neighbour-shed rates. **Kappa itself is never changed**: it is
   the model's fixed global mixing constant ({kappa}); what differs
   between niches is the *content* of the influx (what the neighbours
   shed — i.e. leakage comes from the NEW neighbours, never the old), not
   the mixing rate. Kappa varies only across *models* in the
   sweep-sensitivity analysis, never inside a prediction. **The cell's own
   intrinsic state z is held fixed** — the counterfactual changes the
   context response and the leak source, nothing else. A separate "model
   account" score additionally lets the type's intrinsic mix differ across
   niches; its excess over the counterfactual measures how much of an
   observed niche difference is *selection* rather than context.
5. **The observed shift**: depth-normalised mean expression of the type's
   held-out cells, niche B minus niche A, same log scale; both sides
   mean-centred (softmax normaliser and depth are per-panel constants).
6. **Scores per panel**: calibration slope and R² of predicted vs
   observed across genes, against three references — zero-prediction,
   program-only, leak-only.

**What good looks like**: slope ≈ 1 (predicted shift *sizes* are right,
not just directions); R² as high as Xenium depth allows; and the
pre-registered requirement — **the full model beats both single-channel
references** — because if program-alone sufficed the leak channel is
decoration, and if leak-alone sufficed the "biology" is contamination.

**How to read the summary figure** (below): *left*, mean held-out R² of
the three predictors — good = the full-model bar clearly tallest, with
the beats-both count and slope in the title. *Right*, one point per
panel: x = program-channel (biology) R², y = leak-channel (contamination)
R²; above the dashed diagonal the panel's niche difference is
contamination-dominated. **Position on this map is not good or bad — it
IS the finding** (which niche signatures are biology, which are spillage);
good = many large/bright points (well-predicted panels). The per-panel
scatters (`pair*.png`) show single panels gene-by-gene: good = a tight
cloud on the diagonal.

**Tiering (the honesty guard)**: with data-defined niches,
composition-distinct pairs are disjoint by construction, so the supported
(interpolation) tier is structurally near-empty
({sup.get('n_panels', 0)} panels, mean R²
{sup.get('full', float('nan')):.2f} — adjacent niches, little to
predict) and the informative regime is **extrapolation, named as such**.
Annotation-defined niches (nested bands of the kNN-smoothed tumour
fraction: deep stroma → stroma → interface → rim → core) are *ordered*
and therefore do have shared composition support, which is what populates
the supported tier; they exist only on slides whose type names name a
tumour.

**Two more guards each panel carries.** (i) A **noise ceiling**: the
observed shift is a difference of two sample means over a few hundred
held-out cells, so part of it is sampling noise. Splitting the cells in
half and correlating the two versions of the shift (Spearman–Brown
corrected) gives the largest R² *any* predictor of that panel could
reach. A panel is **trusted** when that ceiling is at least 0.5 on at
least 100 genes; untrusted panels stay in the JSON and are reported
separately, never dropped silently. (ii) A **Φ-fixed row**: the same
counterfactual with the image context Φ frozen at the receiver type's
mean in both niches, so the program channel answers to neighbour
*composition* alone — the part of a neighbourhood an intervention could
actually set. Its R² as a fraction of the real-Φ counterfactual's is the
**interventionable share**.

{table}

**What a pass is, on this run.** The pre-registered bars: the
counterfactual beats both single channels in a *majority* of panels
({ex['full_beats_both']}/{ex['n_panels']} =
{100 * ex['full_beats_both'] / max(ex['n_panels'], 1):.0f} %); median
calibration slope inside [0.8, 1.2] ({ex['median_slope']:.2f}); and, with
annotation niches, at least 20 panels in the supported tier
({band_n} panels). The κ trajectory of program / leak / total belongs
beside this as a κ-*range*, never a point — see
`experiments/transport_kappa_sensitivity_v2.json`.

**What can be concluded, and what cannot.** {conclusion}

{example}![transport summary](../transport/transport_summary.png)
{transport_distribution_block(run_dir)}
"""


def a4_section(run_dir: Path) -> str:
    """Doc-11 A4 (decontaminated cycle call) verdicts, when it has been run."""
    path = run_dir / "applications" / "a4_cycle.json"
    if not path.exists():
        return ""
    res = json.loads(path.read_text())
    st, fp, pl = res["stratified"], res["fingerprint"], res["planted"]
    rows = ["| leg | raw | z | band / rule | reads |",
            "|---|---|---|---|---|",
            f"| exposure gap (Q4 − Q1 call rate) | {st['raw']['gap']:.3f} | "
            f"{st['z']['gap']:.3f} | shuffle [{st['raw']['null_band'][0]:.3f}, "
            f"{st['raw']['null_band'][1]:.3f}] | raw above band = calls track "
            f"exposure; z lower = rejection |",
            f"| gene-split Δ ring 1 (post-mitotic) | "
            f"{fp['post_mitotic']['ring1']['mean']:.4f} "
            f"[{fp['post_mitotic']['ring1']['ci'][0]:.4f}, "
            f"{fp['post_mitotic']['ring1']['ci'][1]:.4f}] | — | ring 2 "
            f"{fp['post_mitotic']['ring2']['mean']:.4f} | > 0 and > ring 2 = "
            f"transcript leak; ≈ 0 = homophily only |",
            f"| planted victim FPR (3 seeds, mean) | "
            f"{np.mean([p['raw']['victim_fpr'] for p in pl]):.2f} | "
            f"{np.mean([p['z']['victim_fpr'] for p in pl]):.2f} | pass = z < raw "
            f"and AUROC kept | {sum(p['pass'] for p in pl)}/3 pass |"]
    verdicts = "\n".join(f"- {v}" for v in res["verdict"])
    return f"""

## A4 — decontaminated cycle call (doc-11)

**How**: calls in the **post-mitotic** types (not the top-4 MKI67 types,
not Unassigned, ≥ 2 000 cells; {res['population']['n_post']} cells in
{len(res['population']['post_mitotic'])} types): raw = Tirosh phase ≠ G1,
z = rate-matched top-k per type by the probe projection max(z·β_S,
z·β_G2M) (β fitted in the cycling types, training folds). Exposure =
β-weighted neighbour cycle score. Three legs: (1) call rate in the top vs
bottom exposure quartile, within type, against a within-type exposure
shuffle band; (2) the gene-split fingerprint — split S+G2M into random
halves A/B, Δ = corr(own_A, nbr_A) − corr(own_A, nbr_B): transcript leak
inflates only the same-half term (Δ > 0, one-hop: ring 1 > ring 2),
niche co-clustering does not (Δ ≈ 0); (3) the planted world (gate
scaffold, κ = 0.2, cycle-like program planted in 30% of two types, leaked
through the true operator) — victim FPR and planted-cell AUROC, raw vs z.
DAPI is kept group-level only (weak ploidy proxy in FFPE sections).
**Evaluated by**: the pre-registered rules in the table. **Wished for**:
if leak-induced positives exist, raw above its band and z below raw with
Δ > 0 one-hop; the planted world must pass regardless (the mechanism
claim). If the real-data legs are null, the honest outcome is "such
false-positives are rare at κ = 0.1 on this slide".

{chr(10).join(rows)}

{verdicts}

![A4 summary](../applications/a4_cycle.png)
"""


def lr_map_section(run_dir: Path) -> str:
    """Doc-09 section 8: the LR co-occurrence map with its two nulls."""
    path = run_dir / "communication" / "lr_map.json"
    if not path.exists():
        return ""
    res = json.loads(path.read_text())
    ap, a, b = res["panel_A_prime"], res["panel_A"], res["panel_B"]
    n = len(res["rows"]) * len(res["columns"])
    return f"""

## LR co-occurrence ladder (doc-09 section 8)

**How**: rows = the effective-rank w programs (rank
{res['effective_rank']['rank']} here; per-cell program coordinates,
labelled by each program's top ± loadings); columns = gate-zero
ligand–receptor pairs ranked by the variance of the composition-
residualised exposure × receiver prevalence, one column per ligand, ≥ 2
receptor-eligible receiver types (top {len(res['columns'])}, CellChatDB
pathway in brackets); column value per cell = log1p of the model's clean
receptor rate × one-hop exposure (receptor side from ρ, mildly circular —
descriptive map). Entry = pooled Spearman, Fisher-z, validation tiles.
Three rungs: **A′** uncontrolled over all cells (the SIMVI-comparable
view), **A** within receiver type, **B** within type after rank-
transforming and ridge-residualising both sides on neighbour
composition. **Evaluated by** two nulls per rung: the doc's permutation of
the LR score (colour; BH q ≤ 0.05, grey = n.s.) and a Moran-preserving
spatial shift of the LR field (boxed entries survive it). Counts below are
heatmap entries (programs × pairs), each a pooled Spearman over all
eligible validation cells. **Wished for**: A′
dense with large |ρ|, A collapsed, B empty — the published-map structure
is type composition; nothing is claimed about signalling.

Permutation null: A′ **{ap['n_significant']} / {n}** (max |ρ|
{ap['max_abs_rho']:.2f}) → A **{a['n_significant']} / {n}** (max |ρ|
{a['max_abs_rho']:.2f}) → B **{b['n_significant']} / {n}** (max |ρ|
{b['max_abs_rho']:.2f}). Spatial-shift null: **{ap['n_significant_shift']} /
{a['n_significant_shift']} / {b['n_significant_shift']}**. {res['caption']}

![LR co-occurrence ladder](../communication/lr_map.png)
"""


def render(run: str, run_meta: dict, config: TrainConfig, metrics: dict,
           history: list[dict], images: dict, pseudo: dict, names: list[str],
           groups) -> str:
    best, final = metrics["best"], metrics["final"]
    cycle = final.get("cycle") or {}
    if "linear_ref" not in cycle and "ceiling" in cycle:
        cycle["linear_ref"] = cycle["ceiling"]      # pre-rename runs
    pool = lambda latent: (cycle.get(latent) or {}).get("r2_pooled",
                                                        float("nan"))
    by_z = (cycle.get("z") or {}).get("by_type") or {}
    by_w = (cycle.get("w") or {}).get("by_type") or {}
    rel = ((cycle.get("reliability") or {}) if isinstance(
        cycle.get("reliability"), dict) else {})
    mirror, probe = final.get("mirror") or {}, final.get("probe") or {}
    fig = lambda key, alt: f"![{alt}](figures/{images.get(key, key + '.png')})"

    return f"""# DisCell run report — `{run}`

Model `{run_meta.get('git', 'unknown')}`, seed {config.seed}, trained
{metrics.get('minutes', float('nan')):.0f} min on `{config.dataset}`
(407k cells, 5.1k genes). Operating point (the calibrated defaults):
κ = {config.kappa}, α_z = {config.alpha_z}, α_w = {config.alpha_w},
invariance = {config.invariance} (α_a = {config.alpha_a},
{config.adv_steps} steps at lr {config.adv_lr}), ω = {config.omega},
d_z = {config.d_z}, d_w = {config.d_w}. Budget: up to {config.epochs}
epochs, early stop after {config.patience} improvement-free epochs; best
checkpoint at epoch {best['epoch']} of {history[-1]['epoch']} trained.

The model splits each cell's expression into an **intrinsic state z**
(what the cell is doing of its own accord), a **spatial response w**
(how its context modulates it, decoded through the gene-programme matrix
B), and a **fixed leakage channel** (κ-mixed neighbour transcripts that
belong to no one's biology). Everything below checks that the split
actually happened.

## Training progress — what to look for

![progress](figures/progress.png)

Each panel is one failure mode being ruled out as training proceeds
(grey line: the best checkpoint under the joint criterion — held-out
reconstruction must improve *and* z-type NMI must not collapse):

- **held-out reconstruction** must rise and plateau; a model can also
  buy reconstruction by leaking, which is why this panel never decides
  alone. Final: **{final['recon_val']:.4f}** per-count nats
  (best {best['recon_val']:.4f} at epoch {best['epoch']}).
- **z-type NMI** flat and high (**{final['nmi']:.3f}**) — the CSVAE
  failure mode is this collapsing while the invariance looks great.
- **probe ΔCE** hugging zero, floor below it
  (**{probe.get('delta_ce', float('nan')):.4f}** vs floor
  {probe.get('noise_floor', float('nan')):.4f}): a fresh ridge probe,
  fitted after the fact, finds no niche information hiding in z.
- **mirror R²** near its permuted control
  (**{mirror.get('r2', float('nan')):.3f}** vs
  {mirror.get('r2_permuted', float('nan')):.3f}): attention is not
  reconstructing the cell through look-alike neighbours.
- **cycle pooled R²** separating: z up, w pinned at zero — the
  disentanglement *emerging* during training, not asserted after it.
- **KL to the priors**: KL_z stays large — z encodes per-cell
  information. KL_w hugs zero, which is the α_w = {config.alpha_w}
  operating point, not a death: the posterior w rides its
  context-conditional prior m_ψ(c,t), so w is *context-determined* by
  construction (and per-cell anomaly readouts are conservative).

## Reconstruction

**How**: mean per-count multinomial log-likelihood of the held-out
validation tiles under the model's leak-mixed rates, evaluated with
posterior means (no sampling noise); the best checkpoint is chosen jointly
with the NMI guard, never on reconstruction alone. **Good**: higher (less
negative). Scale: ±0.005 is seed noise; ±0.01 is a real architectural
effect (the Φ ablation's size).

Held-out per-count log-likelihood **{final['recon_val']:.4f}** nats
(≈ e^{final['recon_val']:.2f} ≈ {np.exp(final['recon_val']):.5f}
multinomial probability per transcript over 5,101 genes; the uniform
baseline is 1/5101 ≈ 0.0002). The number matches the κ = 0.1 stratum of
the sweep grid (−7.254 ± 0.005 over seeds), i.e. the long budget bought
convergence certainty, not a different model.

## The disentanglement quadrant

The core claim is a division of labour, so each channel must *pass its
own* analysis and *fail the other's*. **How**: every cell is a fresh probe
(ridge for continuous targets, principal curves for orderings) fitted
within type on spatially separated folds, and judged against reference
rows — a within-type permutation floor, a log-depth (ℓ) baseline, and for
expression-derived targets a 50-PC expression reference. **Good**: each
target hot in exactly its own column and near the references in the
other; a hot cell only counts if it clears its references. Four cells,
two shown as figures apiece:

| | cell cycle (intrinsic dynamics) | pseudotime (tissue gradient, beyond type) |
|---|---|---|
| **z** | **works**: pooled R² {pool('z'):.2f} | fails: niche R² {pseudo['z']['niche_r2']:.2f}, coherence {pseudo['z']['neighbour_coherence']:.2f} |
| **w** | fails: pooled R² {pool('w'):.2f} | **works**: niche R² {pseudo['w']['niche_r2']:.2f}, coherence {pseudo['w']['neighbour_coherence']:.2f} |

(Cycle references: within-type-permuted control ≈ 0, 50-PC expression
reference {pool('linear_ref'):.2f} — a linear reference line, not a
ceiling: z may legitimately exceed it. The pseudotime columns are
**type-partialled** — z legitimately carries type and type is spatially
predictable through homophily, so raw numbers conflate the two; raw
values appear in the sections below.)

### Cell cycle lives in z…

{fig('z_cycle_projection', 'z cycle projection')}

**Wished for**: pooled within-type R² from z well above the permuted
control and at or above the 50-PC reference — higher means more intrinsic
state retained. Projected onto the probe's own axes (z·β_S vs z·β_G2M),
the cycling types show the expected geometry — a G1 blob at the origin with an arc
through S into G2M. Pooled within-type R² **{pool('z'):.2f}** against the
50-PC linear expression reference {pool('linear_ref'):.2f} (a reference
line, not a ceiling). The *absolute* read is the claim: z far above the
within-type permuted control and the ℓ-baseline, w at zero. The **ratio**
to the linear reference is a property of the target's reliability and of
the frame's strength on this slide, not of the model — it is ≈ 2.1 on the
shallow ovarian and lung FFPE slides, 0.96 on the GSE315411 core and 0.90
on the deep fresh-frozen slide — so it is not quoted as a model property;
the S-side
diffuseness is target noise, not model failure (split-half score
reliability S {rel.get('s', float('nan')):.2f} /
G2M {rel.get('g2m', float('nan')):.2f}).

### …and not in w

![cycle projection z vs w](figures/cycle_projection_zw.png)

**Wished for**: ≈ 0 — any cycle signal in w above the control means
intrinsic state leaked into the context channel. The *identical* probe
construction on w
({', '.join(names[g] for g in groups)}): no arc, no phase separation —
pooled R² **{pool('w'):.2f}**. A faint in-sample trend can appear for
the inflammatory type, whose cycle genuinely co-varies with its
interferon niche gradient, but held-out it evaporates (per-type w R²
{', '.join(f"{by_w.get(str(g), float('nan')):.2f}" for g in groups)} vs
z's {', '.join(f"{by_z.get(str(g), float('nan')):.2f}" for g in groups)}).
If leakage or the GAT had smuggled neighbour state into w,
proliferative neighbourhoods would light this up; they do not.

### The tissue gradient lives in w…

{fig('w_trajectories_umap', 'w trajectories')}

![pseudotime on tissue](figures/pseudotime_tissue.png)

**Wished for**: high niche R² and neighbour coherence — higher means
w's ordering is a genuine tissue gradient. A principal curve fitted in
the full w-space orders cells along a
**niche gradient**: even after removing every per-type mean, neighbour
composition predicts the ordering (held-out R²
**{pseudo['w']['niche_r2']:.2f}**; raw
{pseudo['w']['niche_r2_raw']:.2f}) and graph neighbours agree on it
(coherence **{pseudo['w']['neighbour_coherence']:.2f}**; raw
{pseudo['w']['neighbour_coherence_raw']:.2f}) — painted on the slide it
recovers contiguous tissue domains.

### …and z's ordering is *not* a tissue gradient

{fig('z_trajectories_umap', 'z trajectories')}

**Wished for**: ≈ 0 after type-partialling (the partialling matters:
z legitimately carries type, and type is spatially predictable through
homophily, so raw numbers overstate). The same machinery on z produces a
valid ordering — but of intrinsic state, not space. Its raw niche R² ({pseudo['z']['niche_r2_raw']:.2f})
is mostly type read through homophily: once per-type means are removed,
niche composition explains only **{pseudo['z']['niche_r2']:.2f}** of it
and neighbour coherence drops to
{pseudo['z']['neighbour_coherence']:.2f} (w keeps
{pseudo['w']['neighbour_coherence']:.2f} under the same partialling).
(Per-type, the z axis reads as nameable programmes — tumour-state
continuum, interferon response, contractile↔synthetic; see devlog.)
That is the point: z varies *within* tissue domains, w varies *across*
them.

## Per-type spatial responsiveness — withdrawn as a ranking

{fig('w_norm_by_type', 'per-type w norm boxplot — the gauge, not a score')}

**Withdrawn (issues V12).** Earlier versions of this report ranked types by
‖μ_w‖ and read the ordering biologically. That ranking is **not
identified**: w carries a per-type constant offset, and (a(z) − B·μ_t,
w + μ_t) is the same model for any per-type constant μ_t, so nothing in the
objective fixes it. On the pinned reference the offset is 4–10× the
within-type spread (‖B·mean_t w‖ 18–28), the ranking it produces is the
offset rather than the response (Spearman −0.41 against the same ranking
after centring), and weight decay moves the gauge *into* w instead of
removing it. The figure is kept only to show the quantity that was being
ranked; **no ordering is claimed from it.**

**What replaces it.** The offset-free read of "which types does context
modulate" is the within-type variance of each programme's coordinate on
gauge-centred w, `Δ_i = B·[w_i − w̄_t]` — the *most modulated type* column of
the w-programme atlas section above, where every w read is centred within
type at the reference context. That quantity is invariant to μ_t by
construction and is the one that can be compared across types and across
fits.

## Gene programmes

{fig('B_loadings', 'B loadings')}

**How**: the decoder's B matrix, top genes per column. **Read**:
descriptive only — columns are identified up to an invertible mix, so
compare *gene signatures* across runs, never raw columns; the atlas
section below fixes a canonical basis for exactly this reason.

## Provenance

`config.json`, `history.jsonl` (every evaluation), `metrics.json`, and
the TensorBoard event file live next to this report; the checkpoint is
`best.pt`. Figures above are the final training-time figures plus two
computed fresh from the best checkpoint on the validation split
(`cycle_projection_zw.png`, `pseudotime_tissue.png`).
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(build(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
