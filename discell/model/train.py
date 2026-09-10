#!/usr/bin/env python3
"""One DisCell fit: the loop, its diagnostics, and the TensorBoard record.

One invocation is one point of the kappa sweep. Outputs land under
``data/datasets/<id>/runs/<name>/``: TensorBoard events, ``config.json``,
``best.pt`` (weights + covariance state), ``metrics.json``.

Early stopping is **joint** (spec 7.10): a checkpoint counts as best only when
held-out reconstruction improves *and* z-type NMI has not collapsed below a
fraction of its own running best -- a model explaining everything with leakage
has excellent reconstruction, which is exactly why reconstruction alone must
not decide.

Usage::

    python -m discell.model.train --dataset <id> --embeddings egomask_ego_v1 --kappa 0.2
    python -m discell.model.train --dataset <id> --embeddings egomask_ego_v1 \\
        --kappa 0 --alpha-a 0 --run-name k0_uncontrolled
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from discell import paths
from discell.model import metrics as M
from discell.model.elbo import Weights, adversary_terms, discell_loss
from discell.model.equations import TypeCovariances
from discell.model.networks import DisCell
from discell.model.prepare import ModelData, assemble, tile_batch

log = logging.getLogger("discell.model.train")


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines one fit; serialised into the run directory."""

    dataset: str
    variant: str = "full"
    embeddings: str = "egomask_ego_v1"
    #: obs column for t; None = the bundle default (curated cell_group).
    #: "graphclust" runs on unsupervised clusters -- the no-annotation control
    label_key: str | None = None
    run_name: str | None = None
    # the model
    d_z: int = 20
    d_w: int = 6
    hidden: int = 256
    gat_dim: int = 32
    heads: int = 4
    phi_pca: int | None = None          # None: full-dimension Phi into c
    v_pcs: int = 12                     # Phi PCs inside the invariance block
    # the objective -- defaults are the calibrated operating point (2026-09,
    # four calibration rounds + two sweeps; see docs/devlog.md): adversary at
    # alpha_a = 0.3 / 6 steps / lr 2e-3 (MLP-probe leak 14% of uncontrolled at
    # zero NMI cost), kappa = 0.1 (end of the recon plateau, most B-stable).
    # Under invariance = "closed_form" the straight-through-scaled alpha_a
    # equivalent is ~0.02, not 0.3.
    invariance: str = "adversary"       # "closed_form" before spec 4.6 escalation
    adv_lr: float = 2e-3
    adv_steps: int = 6
    adv_hidden: int = 64
    kappa: float = 0.1
    omega: float = 1.0
    alpha_z: float = 0.007
    alpha_w: float = 0.1
    alpha_a: float = 0.3
    # optimisation
    epochs: int = 200
    tile_cells: int = 4096
    lr: float = 1e-3
    grad_clip: float = 10.0
    val_fraction: float = 0.15
    patience: int = 20
    nmi_guard: float = 0.9              # best must keep NMI >= guard * running max
    cov_ema: float = 0.05
    eval_every: int = 5
    figures_every: int = 25
    panel_types: int = 8            # types shown in per-type figures
    seed: int = 0
    device: str = "cuda"

    def weights(self) -> Weights:
        return Weights(omega=self.omega, alpha_z=self.alpha_z,
                       alpha_w=self.alpha_w, alpha_a=self.alpha_a)

    def name(self) -> str:
        return self.run_name or f"discell_k{self.kappa:g}_seed{self.seed}"


class Trainer:
    """Owns one fit: tiles to device, steps, evaluation sweeps, the log."""

    def __init__(self, config: TrainConfig, data: ModelData) -> None:
        self.config, self.data = config, data
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu"
            else "cpu")
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        # figures draw their own subsamples: a logging knob must never move
        # the training-shuffle stream
        self.figure_rng = np.random.default_rng([config.seed, 3])
        if config.figures_every % config.eval_every:
            raise ValueError("figures_every must be a multiple of eval_every, "
                             f"got {config.figures_every} / {config.eval_every}")

        self.model = DisCell(
            n_genes=data.x.shape[1], n_types=len(data.p_t),
            phi_dim=data.phi.shape[1], median_counts=data.median_counts,
            d_z=config.d_z, d_w=config.d_w, hidden=config.hidden,
            gat_dim=config.gat_dim, heads=config.heads,
        ).to(self.device)
        self.covariances = None
        self.adversary = self.adversary_optimiser = None
        if config.alpha_a and config.invariance == "closed_form":
            self.covariances = TypeCovariances(
                len(data.p_t), config.d_z, data.v_block.shape[1],
                ema=config.cov_ema).to(self.device)
        elif config.alpha_a and config.invariance == "adversary":
            from discell.model.networks import Adversary

            if data.e_phi is None or data.phibar_t is None:
                raise ValueError("invariance='adversary' needs e_phi/phibar_t "
                                 "on the ModelData (assemble builds them)")
            self.adversary = Adversary(config.d_z, len(data.p_t),
                                       data.e_phi.shape[1],
                                       hidden=config.adv_hidden).to(self.device)
            self.adversary_optimiser = torch.optim.Adam(
                self.adversary.parameters(), lr=config.adv_lr)
            self.ybar_t = torch.tensor(data.graph.ybar_t, device=self.device)
            self.phibar_t = torch.tensor(data.phibar_t, device=self.device)
        self.p_t = torch.tensor(data.p_t, device=self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=config.epochs)

        self.train_batches = [self._to_device(t) for t in data.train_tiles]
        self.val_batches = [self._to_device(t) for t in data.val_tiles]

        run_root = paths.dataset(config.dataset).root / "runs"
        self.run_dir = run_root / config.name()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _to_device(self, tile: np.ndarray) -> dict:
        """One tile's tensors, resident on the device for the whole fit."""
        b = tile_batch(self.data.graph, tile)
        dense = torch.tensor(self.data.x[b.nodes].toarray(), device=self.device)
        return dict(
            nodes=b.nodes,
            x=dense,
            t=torch.tensor(self.data.t[b.nodes], device=self.device),
            phi=torch.tensor(self.data.phi[b.nodes], device=self.device),
            isolated=torch.tensor(self.data.graph.isolated[b.nodes],
                                  device=self.device),
            gat_src=torch.tensor(b.gat_src, device=self.device),
            gat_dst=torch.tensor(b.gat_dst, device=self.device),
            leak_src=torch.tensor(b.leak_src, device=self.device),
            leak_dst=torch.tensor(b.leak_dst, device=self.device),
            leak_beta=torch.tensor(b.leak_beta, dtype=torch.float32,
                                   device=self.device),
            v=torch.tensor(self.data.v_block[b.nodes[:b.n_seeds]],
                           device=self.device),
            y_seed=torch.tensor(self.data.graph.y[b.nodes[:b.n_seeds]],
                                device=self.device),
            ephi_seed=None if self.data.e_phi is None else torch.tensor(
                self.data.e_phi[b.nodes[:b.n_seeds]], device=self.device),
            n_seeds=b.n_seeds, n_context=b.n_context,
        )

    @staticmethod
    def _forward_kwargs(batch: dict) -> dict:
        keep = ("x", "t", "phi", "isolated", "gat_src", "gat_dst",
                "leak_src", "leak_dst", "leak_beta", "n_seeds", "n_context")
        return {k: batch[k] for k in keep}

    # -- steps -------------------------------------------------------------

    def _step(self, batch: dict):
        """One model update, then the adversary's update(s) when escalated."""
        import dataclasses as dc

        from discell.model.networks import soft_cross_entropy

        config = self.config
        fwd = self.model(**self._forward_kwargs(batch), kappa=config.kappa)
        n = batch["n_seeds"]
        weights = config.weights()
        extras: dict = {}
        if self.adversary is not None:
            adv = adversary_terms(self.adversary, fwd.mu_z[:n], batch["t"][:n],
                                  batch["y_seed"], batch["ephi_seed"],
                                  self.ybar_t, self.phibar_t)
            terms = discell_loss(fwd, batch["x"][:n], batch["t"][:n],
                                 weights=dc.replace(weights, alpha_a=0.0))
            loss = terms.loss + weights.alpha_a * adv.encoder_term
            terms.penalty = float(adv.encoder_term.detach())
            extras = {"adv_excess_y": adv.excess_y,
                      "adv_excess_phi": adv.excess_phi}
        else:
            terms = discell_loss(fwd, batch["x"][:n], batch["t"][:n],
                                 weights=weights, v=batch["v"],
                                 covariances=self.covariances, p_t=self.p_t)
            loss = terms.loss
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       config.grad_clip)
        self.optimiser.step()

        if self.adversary is not None:
            z_frozen = fwd.mu_z[:n].detach()
            for _ in range(config.adv_steps):
                self.adversary_optimiser.zero_grad()
                log_y, log_phi = self.adversary(z_frozen, batch["t"][:n])
                head_loss = (soft_cross_entropy(batch["y_seed"], log_y)
                             + soft_cross_entropy(batch["ephi_seed"], log_phi)
                             ).mean()
                head_loss.backward()
                self.adversary_optimiser.step()
            extras["adv_head_loss"] = float(head_loss.detach())
        return terms, extras

    @torch.no_grad()
    def _sweep(self, batches: list[dict], want_log_p: bool = False) -> dict:
        """Collect per-seed arrays over *batches* in eval mode."""
        self.model.eval()
        out = {k: [] for k in ("nodes", "mu_z", "mu_w", "prior_w", "c",
                               "kl_w", "log_p")}
        for batch in batches:
            fwd = self.model(**self._forward_kwargs(batch),
                             kappa=self.config.kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(batch["nodes"][:n])
            out["mu_z"].append(fwd.mu_z[:n].cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
            out["c"].append(fwd.c[:n].cpu().numpy())

            per_dim = 0.5 * (-fwd.logvar_w[:n]
                             + fwd.logvar_w[:n].exp()
                             + (fwd.mu_w[:n] - fwd.prior_mean_w[:n]) ** 2 - 1.0)
            out["kl_w"].append(per_dim.cpu().numpy())
            if want_log_p:
                out["log_p"].append(fwd.log_p.cpu().numpy())
        self.model.train()
        return {k: np.concatenate(v) if v else None for k, v in out.items()}

    # -- evaluation --------------------------------------------------------

    def evaluate(self) -> dict:
        val = self._sweep(self.val_batches, want_log_p=True)
        train = self._sweep(self.train_batches)

        x_val = np.vstack([self.data.x[b["nodes"][:b["n_seeds"]]].toarray()
                           for b in self.val_batches])
        rows_all = np.concatenate([train["nodes"], val["nodes"]])
        z_all = np.vstack([train["mu_z"], val["mu_z"]])
        c_all = np.vstack([train["c"], val["c"]])
        t_all = self.data.t[rows_all]

        train_mask = np.zeros(len(rows_all), dtype=bool)
        train_mask[:len(train["nodes"])] = True
        probe = M.probe_delta_ce(z_all, t_all, self.data.v_block[rows_all],
                                 self.data.vbar_t, train_mask, ~train_mask,
                                 seed=self.config.seed)
        cycle = None
        if self.data.cycle is not None:
            cyc = self.data.cycle
            type_names = [str(n) for n in self.data.type_names]
            cycling = np.array([g for g in cyc["cycling_types"]
                                if "nassigned" not in type_names[g]][:4])
            scores = np.stack([cyc["s_score"], cyc["g2m_score"]],
                              axis=1)[rows_all]
            w_all = np.vstack([train["mu_w"], val["mu_w"]])
            cycle = {"types": cycling.tolist(),
                     "reliability": cyc.get("reliability"),
                     "z": M.cycle_r2(z_all, t_all, scores, cycling,
                                     train_mask, ~train_mask,
                                     seed=self.config.seed),
                     "w": M.cycle_r2(w_all, t_all, scores, cycling,
                                     train_mask, ~train_mask,
                                     seed=self.config.seed),
                     # the ceiling: the same probe from 50 expression PCs --
                     # z cannot retain more cycle than the counts carry
                     "ceiling": M.cycle_r2(cyc["x_pcs"][rows_all], t_all,
                                           scores, cycling, train_mask,
                                           ~train_mask,
                                           seed=self.config.seed),
                     # the l-baseline (doc 08 conventions): depth alone --
                     # G2/M cells carry ~2x mRNA, so l is not trivially cold
                     # even for an intrinsic target
                     "lbaseline": M.cycle_r2(
                         np.log(self.data.totals[rows_all].clip(min=1.0)
                                )[:, None].astype(np.float64),
                         t_all, scores, cycling, train_mask, ~train_mask,
                         seed=self.config.seed)}
        return {
            "cycle": cycle,
            "recon_val": M.held_out_reconstruction(x_val, val["log_p"]),
            "nmi": M.z_type_nmi(z_all, t_all, seed=self.config.seed),
            "mirror": M.mirror_r2(z_all, c_all, t_all, seed=self.config.seed),
            "probe": probe,
            "kl_w_per_dim": val["kl_w"].mean(axis=0).tolist(),
            "collected": {"rows": rows_all, "z": z_all,
                          "w": np.vstack([train["mu_w"], val["mu_w"]]),
                          "prior_w": np.vstack([train["prior_w"],
                                                val["prior_w"]]),
                          "kl_w": np.vstack([train["kl_w"], val["kl_w"]])},
        }

    # -- figures -----------------------------------------------------------

    #: preferred types for the per-type panels; falls back to the most
    #: abundant labelled types when a name is absent from this slide.
    CANONICAL_TYPES = ("Tumor Cells", "Macrophages", "T and NK Cells",
                       "Tumor Associated Fibroblasts")

    def _umap(self, points: np.ndarray) -> np.ndarray:
        import umap

        return umap.UMAP(n_neighbors=15, min_dist=0.1,
                         random_state=self.config.seed).fit_transform(points)

    def _panel_types(self) -> list[int]:
        """Canonical types first, filled to ``panel_types`` by abundance."""
        names = [str(n) for n in self.data.type_names]
        chosen = [names.index(n) for n in self.CANONICAL_TYPES if n in names]
        order = np.argsort(-self.data.p_t)
        chosen += [int(g) for g in order
                   if g not in chosen and "nassigned" not in names[g]]
        return chosen[:self.config.panel_types]

    def figures(self, writer, collected: dict, step: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from discell.model.metrics import principal_curve

        rows, z, w = collected["rows"], collected["z"], collected["w"]
        deviation = w - collected["prior_w"]      # what the cell did beyond
        anomaly = collected["kl_w"].sum(axis=1)   # its niche's expectation
        rng = self.figure_rng
        names = [str(n) for n in self.data.type_names]
        t_rows = self.data.t[rows]
        palette = plt.get_cmap("tab20")
        log_w = np.sign(w) * np.log1p(np.abs(w))     # signed log: w is signed
        connected = self.data.graph.degrees[rows] > 0

        # one global subsample; one member set per panel type; every UMAP
        # fitted at most once per figure event via the cache
        pick = rng.choice(len(rows), min(len(rows), 12_000), replace=False)
        umap_cache: dict = {}

        def type_members(g: int) -> np.ndarray:
            members = np.flatnonzero((t_rows == g) & connected)
            member_rng = np.random.default_rng([self.config.seed, 7, g])
            if len(members) > 8_000:
                members = member_rng.choice(members, 8_000, replace=False)
            return members

        members_of = {g: type_members(g) for g in self._panel_types()}

        def project(points: np.ndarray, method: str, key=None) -> np.ndarray:
            if method == "umap":
                if key is not None and key in umap_cache:
                    return umap_cache[key]
                coords = self._umap(points)
                if key is not None:
                    umap_cache[key] = coords
                return coords
            from sklearn.decomposition import PCA

            return PCA(2, random_state=self.config.seed).fit_transform(points)

        # -- spatial grids, one panel per latent dimension -------------------
        def spatial_grid(values: np.ndarray, tag: str) -> None:
            grid_pick = rng.choice(len(rows), min(len(rows), 40_000),
                                   replace=False)
            position = self.data.positions[rows[grid_pick]]
            dims = values.shape[1]
            cols = min(dims, 5)
            nrows = (dims + cols - 1) // cols
            fig, axes = plt.subplots(nrows, cols,
                                     figsize=(3.0 * cols, 3.0 * nrows))
            for k, ax in enumerate(np.ravel(axes)):
                if k >= dims:
                    ax.axis("off")
                    continue
                lim = np.percentile(np.abs(values[grid_pick, k]), 98) or 1.0
                ax.scatter(position[:, 0], position[:, 1],
                           c=values[grid_pick, k], s=0.5, cmap="RdBu_r",
                           vmin=-lim, vmax=lim, rasterized=True)
                ax.set_title(f"{tag}[{k}]", fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"spatial {tag}, posterior mean", fontsize=10)
            writer.add_figure(f"figures/{tag}_spatial", fig, step)

        # -- per-type ||w||: the sweep's headline ranking, live per run ------
        def w_norm_boxplot() -> None:
            norms = np.linalg.norm(w, axis=1)
            groups = [g for g in range(len(self.p_t))
                      if (t_rows == g).sum() >= 30]
            groups.sort(key=lambda g: -norms[t_rows == g].mean())
            fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(groups)), 4.2))
            box = ax.boxplot([norms[t_rows == g] for g in groups],
                             showmeans=True, showfliers=False,
                             patch_artist=True)
            for patch, g in zip(box["boxes"], groups):
                patch.set_facecolor(palette(g % 20)); patch.set_alpha(0.7)
            ax.set_xticklabels([names[g][:18] for g in groups],
                               rotation=45, ha="right", fontsize=6)
            for k, g in enumerate(groups):
                ax.text(k + 1, norms[t_rows == g].mean(),
                        f"{norms[t_rows == g].mean():.1f}",
                        ha="center", va="bottom", fontsize=5)
            ax.set_ylabel("||w||", fontsize=8)
            ax.set_title("spatial responsiveness by type, ordered by mean",
                         fontsize=9)
            fig.tight_layout()
            writer.add_figure("figures/w_norm_by_type", fig, step)

        # -- one figure per VARIABLE: rows = all cells then each panel type;
        # columns = UMAP | PCA (w additionally carries a log-w UMAP column).
        # The all-cells row is coloured by type; type rows by dominant
        # neighbour (z should mix, w should organise).
        def variable_panels(values: np.ndarray, tag: str,
                            extra: tuple[np.ndarray, str] | None = None) -> None:
            columns = [(values, tag, "umap"), (values, tag, "pca")]
            if extra is not None:
                columns.append((extra[0], extra[1], "umap"))
            groups = list(members_of)
            dominant = self.data.graph.y[rows].argmax(axis=1)
            n_rows = 1 + len(groups)
            fig, axes = plt.subplots(n_rows, len(columns),
                                     figsize=(4.0 * len(columns),
                                              3.4 * n_rows))
            axes = np.atleast_2d(axes)
            for c, (source, source_tag, method) in enumerate(columns):
                coords = project(source[pick], method,
                                 key=(source_tag, "global")
                                 if method == "umap" else None)
                ax = axes[0, c]
                for g in range(len(self.p_t)):
                    sel = t_rows[pick] == g
                    if sel.any():
                        ax.scatter(coords[sel, 0], coords[sel, 1], s=0.6,
                                   color=palette(g % 20),
                                   label=names[g][:16], rasterized=True)
                ax.set_title(f"{source_tag} {method.upper()} | all cells",
                             fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                if c == len(columns) - 1:
                    ax.legend(fontsize=4, markerscale=6, ncol=2)
            for r, g in enumerate(groups, start=1):
                members = members_of[g]
                for c, (source, source_tag, method) in enumerate(columns):
                    ax = axes[r, c]
                    if len(members) < 50:
                        ax.axis("off")
                        continue
                    coords = project(source[members], method,
                                     key=(source_tag, g)
                                     if method == "umap" else None)
                    for nb in np.unique(dominant[members]):
                        sel = dominant[members] == nb
                        ax.scatter(coords[sel, 0], coords[sel, 1], s=0.8,
                                   color=palette(int(nb) % 20),
                                   label=names[nb][:14], rasterized=True)
                    ax.set_title(f"{source_tag} {method.upper()} | "
                                 f"{names[g][:22]}", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
                    if c == len(columns) - 1:
                        ax.legend(fontsize=4, markerscale=5, ncol=1,
                                  title="dominant neighbour",
                                  title_fontsize=4)
            fig.suptitle(f"{tag}: all cells by type, then within-type by "
                         "dominant neighbour", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            writer.add_figure(f"figures/{tag}_panels", fig, step)

        # -- z within cycling AND canonical types, coloured by Tirosh phase --
        def z_cycle_panels() -> None:
            """Hard labels default to G1 at Xenium depth, so the quantitative
            read is the continuous-score probe (val/cycle_r2_*); this is the
            picture, in both projections."""
            cyc = self.data.cycle
            phase_colour = {0: "#b8b8b8", 1: "#e6772e", 2: "#7d3ac1"}
            phase_name = {0: "G1", 1: "S", 2: "G2M"}
            wanted = list(dict.fromkeys(
                [g for g in cyc["cycling_types"] if "nassigned" not in names[g]]
                [:4] + list(members_of)))
            groups = [g for g in wanted
                      if ((t_rows == g) & connected).sum() >= 200][:6]
            fig, axes = plt.subplots(len(groups), 2,
                                     figsize=(8.0, 3.4 * len(groups)))
            for r, g in enumerate(groups):
                members = members_of.get(g)
                if members is None:
                    members = type_members(g)
                phase = cyc["phase"][rows[members]]
                for c, method in enumerate(("umap", "pca")):
                    ax = axes[r, c] if len(groups) > 1 else axes[c]
                    coords = project(z[members], method, key=("z", g))
                    for p in (0, 1, 2):
                        sel = phase == p
                        if sel.any():
                            ax.scatter(coords[sel, 0], coords[sel, 1], s=0.8,
                                       color=phase_colour[p],
                                       label=f"{phase_name[p]} ({sel.sum():,})",
                                       rasterized=True)
                    ax.set_title(f"z {method.upper()} | {names[g][:22]}",
                                 fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
                    if c == 1:
                        ax.legend(fontsize=5, markerscale=6)
            fig.suptitle("z within cycling + canonical types, by Tirosh phase "
                         "(hard label defaults to G1 at this depth)",
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure("figures/z_cell_cycle", fig, step)

        # -- pseudotime: curve fitted in FULL latent space, rendered per
        # projection (curve points mapped through their nearest cells), so
        # both projection figures share one pseudotime and differ only in
        # layout. Direction is arbitrary; reads are ordering and spatial
        # organisation, never sign.
        trajectory_cache: dict = {}

        def fitted_trajectory(values: np.ndarray, tag: str, scope) -> tuple:
            key = (tag, scope)
            if key not in trajectory_cache:
                members = pick if scope == "global" else members_of[scope]
                trajectory_cache[key] = principal_curve(values[members])
            return trajectory_cache[key]

        def trajectories(values: np.ndarray, tag: str, method: str) -> None:
            from scipy.spatial import cKDTree

            scopes = [("all cells", "global", pick)]
            scopes += [(names[g], g, members_of[g]) for g in members_of]
            fig, axes = plt.subplots(len(scopes), 2,
                                     figsize=(8.0, 3.4 * len(scopes)))
            for r, (label, scope, members) in enumerate(scopes):
                ax_u, ax_s = axes[r]
                if len(members) < 50:
                    ax_u.axis("off"); ax_s.axis("off")
                    continue
                pseudotime, curve = fitted_trajectory(values, tag, scope)
                coords = project(values[members], method,
                                 key=(tag, scope) if method == "umap" else None)
                _, nearest = cKDTree(values[members]).query(
                    curve, k=min(20, len(members)))
                curve_2d = coords[nearest].mean(axis=1)
                ax_u.scatter(coords[:, 0], coords[:, 1], c=pseudotime, s=0.8,
                             cmap="viridis", rasterized=True)
                ax_u.plot(curve_2d[:, 0], curve_2d[:, 1], color="black", lw=1.2)
                ax_u.set_title(f"{tag} {method.upper()} + curve (fit in "
                               f"{values.shape[1]}-D) | {label[:20]}",
                               fontsize=8)
                position = self.data.positions[rows[members]]
                ax_s.scatter(position[:, 0], position[:, 1], c=pseudotime,
                             s=0.8, cmap="viridis", rasterized=True)
                ax_s.set_aspect("equal")
                ax_s.set_title("pseudotime in tissue coordinates", fontsize=8)
                for ax in (ax_u, ax_s):
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{tag} pseudotime along the full-space principal "
                         "curve (direction arbitrary)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure(f"figures/{tag}_trajectories_{method}", fig, step)

        def w_spatial_extended() -> None:
            """w per dim, its deviation from the prior per dim, then ||w|| and
            the per-cell KL anomaly (spec 4.3: how far this cell deviates from
            its expected response)."""
            grid_pick = rng.choice(len(rows), min(len(rows), 40_000),
                                   replace=False)
            position = self.data.positions[rows[grid_pick]]
            d_w = w.shape[1]
            panels = ([(w[:, k], f"w[{k}]", "RdBu_r", True)
                       for k in range(d_w)]
                      + [(deviation[:, k], f"w[{k}] − m_ψ[{k}]", "RdBu_r", True)
                         for k in range(d_w)]
                      + [(np.linalg.norm(w, axis=1), "‖w‖", "viridis", False),
                         (anomaly, "KL(q(w)‖p)  anomaly", "magma", False)])
            cols = 5
            nrows = (len(panels) + cols - 1) // cols
            fig, axes = plt.subplots(nrows, cols,
                                     figsize=(3.0 * cols, 3.0 * nrows))
            for k, ax in enumerate(np.ravel(axes)):
                if k >= len(panels):
                    ax.axis("off")
                    continue
                values, title, cmap, diverging = panels[k]
                sample = values[grid_pick]
                if diverging:
                    lim = np.percentile(np.abs(sample), 98) or 1.0
                    kwargs = dict(cmap=cmap, vmin=-lim, vmax=lim)
                else:
                    kwargs = dict(cmap=cmap, vmin=np.percentile(sample, 2),
                                  vmax=np.percentile(sample, 98))
                ax.scatter(position[:, 0], position[:, 1], c=sample, s=0.5,
                           rasterized=True, **kwargs)
                ax.set_title(title, fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("spatial w: posterior mean, deviation from the "
                         "prior, magnitude, anomaly", fontsize=10)
            writer.add_figure("figures/w_spatial", fig, step)

        def z_cycle_projection() -> None:
            """The architect's instrument for a low-variance axis: project z
            onto the probe's own directions (z·beta_S vs z·beta_G2M -- expect a
            G1 blob at the origin with an arc through S into G2M), and colour
            the within-type UMAP by kNN-smoothed scores, since the per-cell
            score is mostly noise. UMAP/PCA display dominant variance, not
            retained information; this displays the retained axis directly."""
            from sklearn.neighbors import NearestNeighbors

            cyc = self.data.cycle
            phase_colour = {0: "#b8b8b8", 1: "#e6772e", 2: "#7d3ac1"}
            groups = [g for g in cyc["cycling_types"]
                      if "nassigned" not in names[g]
                      and ((t_rows == g) & connected).sum() >= 300][:4]
            scores = np.stack([cyc["s_score"], cyc["g2m_score"]],
                              axis=1)[rows]
            fig, axes = plt.subplots(len(groups), 3,
                                     figsize=(11.5, 3.4 * len(groups)))
            axes = np.atleast_2d(axes)
            for r, g in enumerate(groups):
                members = members_of.get(g)
                if members is None:
                    members = type_members(g)
                z_c = z[members] - z[members].mean(axis=0)
                target = scores[members] - scores[members].mean(axis=0)
                gram = z_c.T @ z_c + 1e-3 * np.eye(z_c.shape[1])
                beta = np.linalg.solve(gram, z_c.T @ target)   # (d_z, 2)
                projected = z_c @ beta
                phase = cyc["phase"][rows[members]]
                ax = axes[r, 0]
                for p in (0, 1, 2):
                    sel = phase == p
                    if sel.any():
                        ax.scatter(projected[sel, 0], projected[sel, 1],
                                   s=0.9, color=phase_colour[p],
                                   label=("G1", "S", "G2M")[p],
                                   rasterized=True)
                ax.set_title(f"z·β_S vs z·β_G2M | {names[g][:20]}", fontsize=8)
                ax.legend(fontsize=5, markerscale=6)
                neighbours = NearestNeighbors(n_neighbors=min(30, len(members) - 1)
                                              ).fit(z[members])
                nearest = neighbours.kneighbors(return_distance=False)
                smoothed = scores[members][nearest].mean(axis=1)
                coords = project(z[members], "umap", key=("z", g))
                for c, (label, values) in enumerate(
                        (("S (kNN-smoothed)", smoothed[:, 0]),
                         ("G2M (kNN-smoothed)", smoothed[:, 1])), start=1):
                    ax = axes[r, c]
                    lim = (np.percentile(values, 2), np.percentile(values, 98))
                    ax.scatter(coords[:, 0], coords[:, 1], c=values, s=0.9,
                               cmap="magma", vmin=lim[0], vmax=lim[1],
                               rasterized=True)
                    ax.set_title(f"z UMAP, {label}", fontsize=8)
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])
            rel = cyc.get("reliability") or {}
            fig.suptitle("cycle along the probe's own axes (split-half score "
                         f"reliability: S {rel.get('s', float('nan')):.2f}, "
                         f"G2M {rel.get('g2m', float('nan')):.2f})", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure("figures/z_cycle_projection", fig, step)

        w_spatial_extended()
        spatial_grid(z, "z")
        w_norm_boxplot()
        variable_panels(z, "z")
        variable_panels(w, "w", extra=(log_w, "logw"))
        trajectories(w, "w", "umap")
        trajectories(w, "w", "pca")
        trajectories(z, "z", "umap")
        trajectories(z, "z", "pca")
        if self.data.cycle is not None:
            z_cycle_panels()
            z_cycle_projection()

        loadings = self.model.B.weight.detach().cpu().numpy()   # (G, d_w)
        fig, axes = plt.subplots(1, self.config.d_w,
                                 figsize=(3.0 * self.config.d_w, 2.6))
        for k, ax in enumerate(np.atleast_1d(axes)):
            top = np.argsort(-np.abs(loadings[:, k]))[:12]
            ax.barh(range(len(top)), loadings[top, k], height=0.7)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([str(g) for g in top], fontsize=5)
            ax.set_title(f"B[:, {k}] top genes", fontsize=8)
        writer.add_figure("figures/B_loadings", fig, step)
        plt.close("all")

    # -- the fit -----------------------------------------------------------

    def fit(self) -> dict:
        from torch.utils.tensorboard import SummaryWriter

        config = self.config
        writer = SummaryWriter(self.run_dir)
        (self.run_dir / "config.json").write_text(
            json.dumps({**dataclasses.asdict(config),
                        "git": _git_state()}, indent=2))

        history_path = self.run_dir / "history.jsonl"
        history_path.unlink(missing_ok=True)
        best = {"recon_val": -np.inf, "nmi": 0.0, "epoch": -1}
        nmi_max, stale, step = 0.0, 0, 0
        started = time.time()
        for epoch in range(config.epochs):
            for index in self.rng.permutation(len(self.train_batches)):
                terms, extras = self._step(self.train_batches[index])
                if step % 20 == 0:
                    for key, value in terms.scalars().items():
                        writer.add_scalar(f"train/{key}", value, step)
                    for key, value in extras.items():
                        writer.add_scalar(f"train/{key}", value, step)
                    if terms.penalty_info:
                        writer.add_scalar(
                            "train/penalty_excluded_fraction",
                            terms.penalty_info["excluded_fraction"], step)
                step += 1
            self.schedule.step()

            if (epoch + 1) % config.eval_every:
                continue
            report = self.evaluate()
            nmi_max = max(nmi_max, report["nmi"])
            writer.add_scalar("val/recon", report["recon_val"], step)
            writer.add_scalar("val/nmi", report["nmi"], step)
            writer.add_scalar("val/mirror_r2", report["mirror"]["r2"], step)
            writer.add_scalar("val/mirror_r2_permuted",
                              report["mirror"]["r2_permuted"], step)
            writer.add_scalar("val/probe_delta_ce",
                              report["probe"]["delta_ce"], step)
            writer.add_scalar("val/probe_noise_floor",
                              report["probe"]["noise_floor"], step)
            if report.get("cycle"):
                type_names = [str(n) for n in self.data.type_names]
                for latent in ("z", "w", "ceiling", "lbaseline"):
                    entry = report["cycle"][latent]
                    writer.add_scalar(f"val/cycle_r2_{latent}",
                                      entry["r2_mean_types"], step)
                    writer.add_scalar(f"val/cycle_r2_{latent}_pooled",
                                      entry["r2_pooled"], step)
                    writer.add_scalar(f"val/cycle_r2_{latent}_permuted",
                                      entry["r2_permuted"], step)
                    for g, value in entry["by_type"].items():
                        writer.add_scalar(
                            f"val/cycle_r2_{latent}_types/{type_names[g][:24]}",
                            value, step)
            for k, value in enumerate(report["kl_w_per_dim"]):
                writer.add_scalar(f"val/kl_w_dim{k}", value, step)
            with history_path.open("a") as sink:
                sink.write(json.dumps(
                    {"epoch": epoch, "step": step,
                     **{k: v for k, v in report.items() if k != "collected"}},
                    default=float) + "\n")
            log.info("epoch %d  recon %.4f  nmi %.3f  mirror %.3f  dCE %.4f  [%.0fs]",
                     epoch, report["recon_val"], report["nmi"],
                     report["mirror"]["r2"], report["probe"]["delta_ce"],
                     time.time() - started)

            if (epoch + 1) % config.figures_every == 0:
                self.figures(writer, report["collected"], step)

            improved = report["recon_val"] > best["recon_val"]
            guarded = report["nmi"] >= config.nmi_guard * nmi_max
            if improved and guarded:
                best = {"recon_val": report["recon_val"], "nmi": report["nmi"],
                        "epoch": epoch}
                stale = 0
                covariance_state = None
                if self.covariances is not None:
                    covariance_state = {
                        name: getattr(self.covariances, name).cpu()
                        for name in ("mean", "second", "batch_count", "seen")}
                torch.save({"model": self.model.state_dict(),
                            "covariances": covariance_state,
                            "config": dataclasses.asdict(config),
                            "epoch": epoch}, self.run_dir / "best.pt")
            else:
                stale += 1
                if stale >= max(1, config.patience // config.eval_every):
                    log.info("early stop at epoch %d (best %d)", epoch, best["epoch"])
                    break

        final = self.evaluate()
        final.pop("collected")
        summary = {"best": best, "final": final,
                   "minutes": (time.time() - started) / 60}
        (self.run_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str))
        writer.close()
        log.info("run %s: best recon %.4f (epoch %d), NMI %.3f, %.1f min",
                 config.name(), best["recon_val"], best["epoch"], best["nmi"],
                 summary["minutes"])
        return summary


def _git_state() -> str:
    """Best-effort commit id (+dirty marker) so a run names the code it ran."""
    import subprocess

    try:
        root = Path(__file__).resolve().parents[2]
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout
        return f"{head}{'+dirty' if dirty.strip() else ''}" if head else "unknown"
    except Exception:
        return "unknown"


def run(config: TrainConfig) -> dict:
    data = assemble(config.dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    return Trainer(config, data).fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    defaults = TrainConfig(dataset="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default=defaults.variant)
    parser.add_argument("--embeddings", default=defaults.embeddings)
    parser.add_argument("--run-name", default=None)
    for field in ("kappa", "omega", "alpha_z", "alpha_w", "alpha_a", "lr",
                  "adv_lr", "val_fraction", "nmi_guard", "cov_ema", "grad_clip"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=float,
                            default=getattr(defaults, field))
    for field in ("d_z", "d_w", "hidden", "gat_dim", "heads", "adv_steps",
                  "adv_hidden", "v_pcs", "epochs", "tile_cells", "patience",
                  "eval_every", "figures_every", "panel_types", "seed"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=int,
                            default=getattr(defaults, field))
    parser.add_argument("--phi-pca", type=int, default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--invariance", default=defaults.invariance,
                        choices=("closed_form", "adversary"))
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = vars(build_parser().parse_args(argv))
    quiet = args.pop("quiet")
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    run(TrainConfig(**args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
