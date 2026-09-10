#!/usr/bin/env python3
"""The communication experiment (doc 09): w <-> cell-cell signalling.

Does w carry ligand-induced response -- connected to genes through B --
while z does not, and can the kappa-channel separate communication from its
near-perfect mimic, leakage? The treatment is OBSERVED: neighbour ligand
expression through the model's own graph weights, never the cell's own
counts. Conventions inherit from doc 08 (posterior means, within receiver
type, spatial-block CV, floor / l / y baselines).

Certification is asymmetric by design (doc 09 section 3.1): z-cold is the
allegiance claim; w-hot alone is only "exposure visibility" (the GAT sees
sender z), so the induced-response claim rides on the NicheNet-target AUROC
of the program B theta and on the decoy control -- never on the ridge R2.

Choices the doc leaves open (each logged in the devlog):
- sender types: per-type mean of x_L/l in the top quartile across types;
- receiver types: >= 2000 cells, every receptor subunit detected in >= 5%;
- NicheNet targets: top 250 by regulatory potential, intersected with panel;
- program ranking: |B theta| over genes expressed in >= 1% of cells, the
  ligand itself and receptor subunits excluded;
- matched-null ligands: 50 nearest in (in-panel target count, mean target
  prevalence), scored on the SAME ranking;
- naive hits: BH-FDR < 0.05 partial correlation (x_g ~ E-tilde | log l);
- leak-attributed: the association on decontaminated log rho keeps < 30%
  of the naive coefficient; w-retained: gene in the top-100 |B theta|.

Usage::

    python -m discell.model.communication --dataset <id> --run alphaw_0.03
    python -m discell.model.communication --dataset <id> --run reference_best \\
        --pairs 10 --stages allegiance
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
from discell.model.prepare import ModelData
from discell.model.validate import (collect_latents, load_run, ridge_cv, r2)

log = logging.getLogger("discell.model.communication")

EXTERNAL = Path("data/external")
N_TARGETS = 250            #: NicheNet targets per ligand before panel cut
MIN_TARGETS_IN_PANEL = 20
MIN_RECEIVER_CELLS = 2_000
RECEPTOR_POSITIVE_FRACTION = 0.05
EXPOSURE_VAR_RATIO = 0.1   #: below this, no within-composition variation
MID_BAND_UM = (50.0, 150.0)
MAX_CELLS = 30_000


# -- section 1: pair inventory ----------------------------------------------

def load_databases(panel: set[str]) -> tuple[list[dict], "object"]:
    """CellChat pairs (secreted + contact) with in-panel NicheNet targets.

    Returns the surviving pairs and the NicheNet matrix (for matched nulls).
    Provenance: CellChatDB.human.rda (jinworks/CellChat) and the v2
    nsga2r ligand-target matrix (Zenodo 10.5281/zenodo.7074291).
    """
    import pyreadr
    import rdata

    db = rdata.read_rda(EXTERNAL / "CellChatDB.human.rda")["CellChatDB.human"]
    inter, complexes = db["interaction"], db["complex"]
    nichenet = pyreadr.read_r(
        EXTERNAL / "nichenet_ligand_target_matrix_nsga2r_final.rds")[None]

    def subunits(name: str) -> list[str]:
        if name in complexes.index:
            return [s for s in complexes.loc[name]
                    if isinstance(s, str) and s]
        return [name]

    pairs = []
    keep = inter[inter["annotation"].isin(["Secreted Signaling",
                                           "Cell-Cell Contact"])]
    for _, row in keep.iterrows():
        lig_subs = subunits(row["ligand"])
        rec_subs = subunits(row["receptor"])
        if not all(g in panel for g in lig_subs + rec_subs):
            continue
        ligand = lig_subs[0]
        if ligand not in nichenet.columns:
            continue
        targets = set(nichenet[ligand].nlargest(N_TARGETS).index) & panel
        if len(targets) < MIN_TARGETS_IN_PANEL:
            continue
        pairs.append({"name": str(row["interaction_name"]),
                      "ligand": ligand, "ligand_subunits": lig_subs,
                      "receptor_subunits": rec_subs,
                      "secreted": row["annotation"] == "Secreted Signaling",
                      "targets": sorted(targets)})
    log.info("gate zero: %d pairs (%d secreted), %d unique ligands",
             len(pairs), sum(p["secreted"] for p in pairs),
             len({p["ligand"] for p in pairs}))
    return pairs, nichenet


def sender_receiver_types(data: ModelData, x_rate, pairs: list[dict]) -> None:
    """Attach sender/receiver type lists to each pair, in place."""
    names = [str(n) for n in data.type_names]
    gene_of = {g: k for k, g in enumerate(np.asarray(data.gene_names))}
    counts = np.bincount(data.t, minlength=len(names))
    for pair in pairs:
        lig_col = np.asarray(x_rate[:, gene_of[pair["ligand"]]].todense()
                             ).ravel()
        type_means = np.array([lig_col[data.t == g].mean()
                               for g in range(len(names))])
        threshold = np.quantile(type_means, 0.75)
        pair["sender_types"] = [g for g in range(len(names))
                                if type_means[g] >= threshold
                                and type_means[g] > 0]
        receptor_cols = [np.asarray((x_rate[:, gene_of[s]] > 0).todense()
                                    ).ravel()
                         for s in pair["receptor_subunits"]]
        pair["receiver_types"] = [
            g for g in range(len(names))
            if counts[g] >= MIN_RECEIVER_CELLS
            and all(col[data.t == g].mean() >= RECEPTOR_POSITIVE_FRACTION
                    for col in receptor_cols)
            and g not in pair["sender_types"]]


# -- section 2: exposure -----------------------------------------------------

def exposure_onehop(data: ModelData, gene_column: np.ndarray) -> np.ndarray:
    """``E_i = sum_j beta_ij x_jL / l_j`` -- the doc's formula verbatim."""
    return np.asarray(data.graph.in_edges @ gene_column).ravel()


def exposure_midband(data: ModelData, gene_column: np.ndarray,
                     grid_um: float = 10.0) -> np.ndarray:
    """Gaussian-shell average of ``x_L/l`` over 50-150 um, via a grid KDE.

    Kernel: radial Gaussian centred mid-band (100 um, sigma 30) zeroed
    outside the shell; the field is normalised by the same-kernel density so
    the value is an average, not a sum (dense regions do not inflate it).
    """
    from scipy.signal import fftconvolve

    lo, hi = MID_BAND_UM
    pos = data.positions
    origin = pos.min(axis=0) - grid_um
    ij = np.floor((pos - origin) / grid_um).astype(int)
    shape = ij.max(axis=0) + 2
    mass = np.zeros(shape); density = np.zeros(shape)
    np.add.at(mass, (ij[:, 0], ij[:, 1]), gene_column)
    np.add.at(density, (ij[:, 0], ij[:, 1]), 1.0)

    radius_px = int(np.ceil(hi / grid_um))
    axis = np.arange(-radius_px, radius_px + 1) * grid_um
    dist = np.hypot(axis[:, None], axis[None, :])
    kernel = np.exp(-((dist - 100.0) ** 2) / (2 * 30.0 ** 2))
    kernel[(dist < lo) | (dist > hi)] = 0.0

    field = fftconvolve(mass, kernel, mode="same")
    weight = fftconvolve(density, kernel, mode="same").clip(min=1e-9)
    return (field / weight)[ij[:, 0], ij[:, 1]]


def residualise(exposure: np.ndarray, data: ModelData,
                members: np.ndarray) -> tuple[np.ndarray, float]:
    """``E-tilde = E - ridge(E ~ y)`` within one receiver type's cells.

    The load-bearing step: a ligand marks its sender type, so raw E is
    sender proximity; the claim rides on ligand variation GIVEN composition.
    """
    y = data.graph.y[members]
    design = np.hstack([y - y.mean(axis=0), np.ones((len(members), 1))])
    target = exposure[members]
    gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
    fitted = design @ np.linalg.solve(gram, design.T @ target)
    residual = target - fitted
    ratio = float(residual.var() / max(target.var(), 1e-12))
    return residual, ratio


# -- section 3: allegiance and gene program ---------------------------------

def matched_null_ligands(nichenet, ligand: str, panel: set[str],
                         prevalence: dict, k: int = 50) -> list[str]:
    """Ligands matched on in-panel target count and mean target prevalence."""
    stats = {}
    for cand in nichenet.columns:
        targets = set(nichenet[cand].nlargest(N_TARGETS).index) & panel
        if len(targets) < MIN_TARGETS_IN_PANEL:
            continue
        stats[cand] = (len(targets),
                       float(np.mean([prevalence[g] for g in targets])))
    ref = np.array(stats[ligand])
    scale = np.array([np.std([v[0] for v in stats.values()]) + 1e-9,
                      np.std([v[1] for v in stats.values()]) + 1e-9])
    ranked = sorted((np.abs((np.array(v) - ref) / scale).sum(), cand)
                    for cand, v in stats.items() if cand != ligand)
    return [cand for _, cand in ranked[:k]]


def target_auroc(ranking: np.ndarray, is_target: np.ndarray) -> float:
    """AUROC of target membership in a score ranking (higher = hit)."""
    from sklearn.metrics import roc_auc_score

    if is_target.sum() < 3 or is_target.all():
        return float("nan")
    return float(roc_auc_score(is_target, ranking))


def pair_allegiance(pair: dict, data: ModelData, latents: dict,
                    b_matrix: np.ndarray, exposure: np.ndarray,
                    gene_names: np.ndarray, expressed: np.ndarray,
                    rng) -> dict:
    """Ridge E-tilde from every latent + references; theta -> program."""
    out: dict = {"receiver_types": {}, "theta_by_type": {}}
    names = [str(n) for n in data.type_names]
    log_l = np.log(data.totals.clip(min=1.0))[:, None]
    for g in pair["receiver_types"]:
        members = np.flatnonzero((data.t == g) & (data.graph.degrees > 0))
        if len(members) > MAX_CELLS:
            members = np.sort(rng.choice(members, MAX_CELLS, replace=False))
        e_tilde, ratio = residualise(exposure, data, members)
        if ratio < EXPOSURE_VAR_RATIO:
            out["receiver_types"][names[g]] = {"dropped": True,
                                               "var_ratio": ratio}
            continue
        fold = latents["fold"][members]
        if len(np.unique(fold)) < 2:
            continue
        floor = e_tilde[rng.permutation(len(members))]
        rows = {"var_ratio": ratio, "n_cells": int(len(members))}
        for label, design in (
                ("w", latents["mu_w"][members]),
                ("z", latents["mu_z"][members]),
                ("lbaseline", log_l[members]),
                ("ybaseline", data.graph.y[members])):
            rows[label] = r2(e_tilde, ridge_cv(design, e_tilde, fold))
        rows["floor"] = r2(floor, ridge_cv(latents["mu_w"][members],
                                           floor, fold))
        out["receiver_types"][names[g]] = rows

        w_members = latents["mu_w"][members]
        centred = w_members - w_members.mean(axis=0)
        gram = centred.T @ centred + 1e-3 * np.eye(centred.shape[1])
        theta = np.linalg.solve(gram, centred.T @ (e_tilde - e_tilde.mean()))
        out["theta_by_type"][names[g]] = (
            theta / max(np.linalg.norm(theta), 1e-12)).tolist()

    thetas = np.array(list(out["theta_by_type"].values()))
    if len(thetas):
        mean_theta = thetas.mean(axis=0)
        mean_theta /= max(np.linalg.norm(mean_theta), 1e-12)
        loading = b_matrix @ mean_theta
        excluded = set(pair["ligand_subunits"] + pair["receptor_subunits"])
        usable = expressed & ~np.isin(gene_names, sorted(excluded))
        ranking = np.abs(loading)[usable]
        in_targets = np.isin(gene_names[usable], pair["targets"])
        out["theta"] = mean_theta.tolist()
        out["program_top"] = [
            (str(gene_names[i]), float(loading[i]))
            for i in np.argsort(-np.abs(np.where(usable, loading, 0)))[:15]]
        out["target_auroc"] = target_auroc(ranking, in_targets)
        out["usable_mask"] = usable          # consumed by the null; stripped
        out["ranking"] = ranking
    return out


# -- section 4: leak reattribution ------------------------------------------

def partial_association(x_block: np.ndarray, e_tilde: np.ndarray,
                        log_l: np.ndarray) -> np.ndarray:
    """Per-gene partial correlation with the exposure, controlling depth."""
    def residual(values):
        design = np.stack([np.ones(len(log_l)), log_l], axis=1)
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coef

    x_res = residual(x_block)
    e_res = residual(e_tilde)
    x_res /= (x_res.std(axis=0) + 1e-12)
    e_res /= (e_res.std() + 1e-12)
    return (x_res * e_res[:, None]).mean(axis=0)


def reattribution(pair: dict, data: ModelData, latents: dict,
                  log_rho: np.ndarray, rows_of: np.ndarray,
                  exposure: np.ndarray, gene_names: np.ndarray,
                  program_top: set[str], sender_distance: np.ndarray,
                  rng) -> dict:
    """Naive exposure-DE, then: does each hit survive on decontaminated rho?"""
    out = {}
    for g in pair["receiver_types"]:
        members = np.flatnonzero((data.t == g) & (data.graph.degrees > 0))
        if len(members) > MAX_CELLS:
            members = np.sort(rng.choice(members, MAX_CELLS, replace=False))
        e_tilde, ratio = residualise(exposure, data, members)
        if ratio < EXPOSURE_VAR_RATIO:
            continue
        log_l = np.log(data.totals[members].clip(min=1.0))
        x_norm = np.log1p(np.asarray(data.x[members].todense())
                          / data.totals[members][:, None]
                          * data.median_counts)
        naive = partial_association(x_norm, e_tilde, log_l)
        n = len(members)
        from scipy.stats import norm
        p = 2 * norm.sf(np.abs(naive) * np.sqrt(max(n - 3, 1)))
        # Benjamini-Hochberg at 0.05
        order = np.argsort(p)
        below = p[order] <= 0.05 * (np.arange(len(p)) + 1) / len(p)
        cutoff = order[:np.flatnonzero(below).max() + 1] \
            if below.any() else np.array([], dtype=int)
        significant = np.zeros(len(p), dtype=bool)
        significant[cutoff] = True
        significant &= np.abs(naive) >= 0.05
        hits = np.flatnonzero(significant)
        if not len(hits):
            out[str(data.type_names[g])] = {"n_hits": 0}
            continue

        rho_block = log_rho[rows_of[members]][:, hits].astype(np.float32)
        on_rho = partial_association(rho_block, e_tilde, log_l)
        keeps = np.abs(on_rho) >= 0.3 * np.abs(naive[hits])
        hit_names = gene_names[hits]
        in_program = np.isin(hit_names, sorted(program_top))
        # artifact candidates: the ligand + sender-exclusive markers
        # (slide-internal: expressed in senders, absent in far receivers)
        far = members[sender_distance[members] > 50.0]
        artifact = np.zeros(len(hits), dtype=bool)
        if len(far) >= 200:
            far_prev = np.asarray((data.x[far][:, hits] > 0).mean(axis=0)
                                  ).ravel()
            sender_cells = np.flatnonzero(np.isin(data.t,
                                                  pair["sender_types"]))
            sender_prev = np.asarray(
                (data.x[sender_cells][:, hits] > 0).mean(axis=0)).ravel()
            artifact = (sender_prev >= 0.2) & (far_prev < 0.02)
        artifact |= np.isin(hit_names, pair["ligand_subunits"])

        out[str(data.type_names[g])] = {
            "n_hits": int(len(hits)),
            "leak_attributed": int((~keeps).sum()),
            "w_retained": int((keeps & in_program).sum()),
            "unexplained": int((keeps & ~in_program).sum()),
            "artifact_candidates_among_hits": int(artifact.sum()),
            "artifact_leak_attributed": int((artifact & ~keeps).sum()),
            "examples_leak": [str(n) for n in hit_names[~keeps][:8]],
            "examples_retained": [str(n)
                                  for n in hit_names[keeps & in_program][:8]],
        }
    return out


# -- orchestration ----------------------------------------------------------

def sweep_log_rho(trainer, data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    """Decontaminated log rates for every cell (posterior means)."""
    import torch

    chunks, nodes = [], []
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            chunks.append(fwd.log_rho[:n].cpu().numpy().astype(np.float16))
            nodes.append(batch["nodes"][:n])
    nodes = np.concatenate(nodes)
    order = np.argsort(nodes)
    return np.concatenate(chunks)[order], nodes[order]


def run_experiment(args: argparse.Namespace) -> dict:
    from scipy.spatial import cKDTree

    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    rng = np.random.default_rng(config.seed)
    latents = collect_latents(trainer, data)
    gene_names = np.asarray([str(g) for g in data.gene_names])
    panel = set(gene_names)
    gene_of = {g: k for k, g in enumerate(gene_names)}
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsc()
    prevalence_arr = np.asarray((data.x > 0).mean(axis=0)).ravel()
    prevalence = dict(zip(gene_names, prevalence_arr))
    expressed = prevalence_arr >= 0.01

    pairs, nichenet = load_databases(panel)
    sender_receiver_types(data, x_rate, pairs)
    pairs = [p for p in pairs if p["sender_types"] and p["receiver_types"]]

    # rank by Var(residualised exposure) x receiver prevalence, keep top-k
    for pair in pairs:
        lig_col = np.asarray(x_rate[:, gene_of[pair["ligand"]]].todense()
                             ).ravel()
        pair["exposure"] = exposure_onehop(data, lig_col)
        spread, weight = 0.0, 0
        for g in pair["receiver_types"]:
            members = np.flatnonzero((data.t == g)
                                     & (data.graph.degrees > 0))
            if len(members) > MAX_CELLS:   # random, never first-N (issues T7)
                members = np.sort(rng.choice(members, MAX_CELLS,
                                             replace=False))
            e_tilde, ratio = residualise(pair["exposure"], data, members)
            if ratio >= EXPOSURE_VAR_RATIO:
                spread += e_tilde.var() * len(members); weight += len(members)
        pair["rank_score"] = spread * (weight / data.graph.n_cells) \
            if weight else 0.0
    pairs = sorted(pairs, key=lambda p: -p["rank_score"])[:args.pairs]
    log.info("kept %d pairs after exposure ranking", len(pairs))

    log_rho = rows_of = sender_distance = None
    if "reattribution" in args.stages:
        log_rho, rho_nodes = sweep_log_rho(trainer, data)
        rows_of = np.empty(data.graph.n_cells, dtype=np.int64)
        rows_of[rho_nodes] = np.arange(len(rho_nodes))

    results: dict = {"run": args.run, "alpha_w": config.alpha_w,
                     "kappa": config.kappa, "pairs": {}}
    for pair in pairs:
        entry: dict = {"ligand": pair["ligand"],
                       "receptors": pair["receptor_subunits"],
                       "secreted": pair["secreted"],
                       "sender_types": [str(data.type_names[g])
                                        for g in pair["sender_types"]],
                       "n_targets": len(pair["targets"])}
        allegiance = pair_allegiance(pair, data, latents, b_matrix,
                                     pair["exposure"], gene_names,
                                     expressed, rng)
        if "ranking" in allegiance:
            nulls = []
            usable, ranking = allegiance.pop("usable_mask"), \
                allegiance.pop("ranking")
            for cand in matched_null_ligands(nichenet, pair["ligand"],
                                             panel, prevalence,
                                             k=args.null_ligands):
                cand_targets = set(nichenet[cand].nlargest(N_TARGETS).index) \
                    & panel
                nulls.append(target_auroc(
                    ranking, np.isin(gene_names[usable],
                                     sorted(cand_targets))))
            nulls = [v for v in nulls if np.isfinite(v)]
            if nulls and np.isfinite(allegiance.get("target_auroc",
                                                    float("nan"))):
                allegiance["null_auroc_mean"] = float(np.mean(nulls))
                allegiance["null_percentile"] = float(np.mean(
                    allegiance["target_auroc"] > np.array(nulls)))
        entry["allegiance"] = allegiance

        if "reattribution" in args.stages and "theta" in allegiance:
            senders = np.flatnonzero(np.isin(data.t, pair["sender_types"]))
            sender_distance = cKDTree(data.positions[senders]).query(
                data.positions)[0]
            program_top = {name for name, _ in allegiance["program_top"]} | {
                str(gene_names[i]) for i in np.argsort(
                    -np.abs(b_matrix @ np.array(allegiance["theta"])))[:100]}
            entry["reattribution"] = reattribution(
                pair, data, latents, log_rho, rows_of, pair["exposure"],
                gene_names, program_top, sender_distance, rng)

        if "controls" in args.stages and "theta" in allegiance:
            # decoy exposure: a sender-expressed non-ligand, non-target gene
            sender_cells = np.flatnonzero(np.isin(data.t,
                                                  pair["sender_types"]))
            lig_level = float(np.asarray(
                x_rate[sender_cells][:, gene_of[pair["ligand"]]].todense()
                ).mean())
            all_ligands = {p["ligand"] for p in pairs}
            candidates = [g for g in gene_names
                          if g not in all_ligands
                          and g not in pair["targets"]
                          and prevalence[g] >= 0.05]
            sampled = rng.choice(candidates,
                                 min(200, len(candidates)), replace=False)
            levels = np.array([np.asarray(
                x_rate[sender_cells][:, gene_of[g]].todense()).mean()
                for g in sampled])
            chosen = sampled[np.argmin(np.abs(levels - lig_level))]
            decoy_col = np.asarray(x_rate[:, gene_of[chosen]].todense()
                                   ).ravel()
            decoy_pair = dict(pair)
            decoy_pair["exposure"] = exposure_onehop(data, decoy_col)
            decoy = pair_allegiance(decoy_pair, data, latents, b_matrix,
                                    decoy_pair["exposure"], gene_names,
                                    expressed, rng)
            decoy.pop("usable_mask", None); decoy.pop("ranking", None)
            entry["decoy"] = {"gene": str(chosen),
                              "target_auroc": decoy.get("target_auroc")}

        results["pairs"][pair["name"]] = entry
        log.info("pair %s: auroc %s (null %s)", pair["name"],
                 allegiance.get("target_auroc"),
                 allegiance.get("null_auroc_mean"))

    out_dir = run_dir / "communication"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "communication.json").write_text(
        json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out_dir / "communication.json")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--null-ligands", type=int, default=50)
    parser.add_argument("--stages",
                        default="allegiance,reattribution,controls")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.stages = args.stages.split(",")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run_experiment(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
