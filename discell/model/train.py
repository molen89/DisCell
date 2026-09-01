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
from discell.model.elbo import Weights, discell_loss
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
    run_name: str | None = None
    # the model
    d_z: int = 20
    d_w: int = 6
    hidden: int = 256
    gat_dim: int = 32
    heads: int = 4
    phi_pca: int | None = None          # None: full-dimension Phi into c
    v_pcs: int = 12                     # Phi PCs inside the invariance block
    # the objective
    kappa: float = 0.0
    omega: float = 1.0
    alpha_z: float = 0.007
    alpha_w: float = 0.1
    # rescaled when the penalty gradient became straight-through: the old 0.3
    # was silently attenuated by cov_ema (~x0.05), so ~0.015 was its true size
    alpha_a: float = 0.02
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
        self.covariances = TypeCovariances(
            len(data.p_t), config.d_z, data.v_block.shape[1],
            ema=config.cov_ema).to(self.device) if config.alpha_a else None
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
            n_seeds=b.n_seeds, n_context=b.n_context,
        )

    @staticmethod
    def _forward_kwargs(batch: dict) -> dict:
        keep = ("x", "t", "phi", "isolated", "gat_src", "gat_dst",
                "leak_src", "leak_dst", "leak_beta", "n_seeds", "n_context")
        return {k: batch[k] for k in keep}

    # -- steps -------------------------------------------------------------

    def _step(self, batch: dict):
        fwd = self.model(**self._forward_kwargs(batch), kappa=self.config.kappa)
        n = batch["n_seeds"]
        terms = discell_loss(fwd, batch["x"][:n], batch["t"][:n],
                             weights=self.config.weights(), v=batch["v"],
                             covariances=self.covariances, p_t=self.p_t)
        self.optimiser.zero_grad()
        terms.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.config.grad_clip)
        self.optimiser.step()
        return terms

    @torch.no_grad()
    def _sweep(self, batches: list[dict], want_log_p: bool = False) -> dict:
        """Collect per-seed arrays over *batches* in eval mode."""
        self.model.eval()
        out = {k: [] for k in ("nodes", "mu_z", "mu_w", "c", "kl_w", "log_p")}
        for batch in batches:
            fwd = self.model(**self._forward_kwargs(batch),
                             kappa=self.config.kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(batch["nodes"][:n])
            out["mu_z"].append(fwd.mu_z[:n].cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
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
        return {
            "recon_val": M.held_out_reconstruction(x_val, val["log_p"]),
            "nmi": M.z_type_nmi(z_all, t_all, seed=self.config.seed),
            "mirror": M.mirror_r2(z_all, c_all, t_all, seed=self.config.seed),
            "probe": probe,
            "kl_w_per_dim": val["kl_w"].mean(axis=0).tolist(),
            "collected": {"rows": rows_all, "z": z_all,
                          "w": np.vstack([train["mu_w"], val["mu_w"]])},
        }

    # -- figures -----------------------------------------------------------

    def figures(self, writer, collected: dict, step: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, z, w = collected["rows"], collected["z"], collected["w"]
        pick = self.figure_rng.choice(len(rows), min(len(rows), 40_000),
                                      replace=False)
        position = self.data.positions[rows[pick]]

        fig, axes = plt.subplots(1, self.config.d_w,
                                 figsize=(3.2 * self.config.d_w, 3.2))
        for k, ax in enumerate(np.atleast_1d(axes)):
            lim = np.percentile(np.abs(w[pick, k]), 98) or 1.0
            ax.scatter(position[:, 0], position[:, 1], c=w[pick, k], s=0.5,
                       cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
            ax.set_title(f"w[{k}]", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("spatial response, posterior mean", fontsize=10)
        writer.add_figure("figures/w_spatial", fig, step)

        from sklearn.decomposition import PCA

        coords = PCA(2).fit_transform(z[pick])
        fig, ax = plt.subplots(figsize=(5, 4))
        for g in range(len(self.p_t)):
            sel = self.data.t[rows[pick]] == g
            if sel.any():
                ax.scatter(coords[sel, 0], coords[sel, 1], s=0.5,
                           label=str(self.data.type_names[g])[:16],
                           rasterized=True)
        ax.legend(fontsize=4, markerscale=6, ncol=2)
        ax.set_title("z, first two PCs, by type", fontsize=9)
        writer.add_figure("figures/z_pca", fig, step)

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

        best = {"recon_val": -np.inf, "nmi": 0.0, "epoch": -1}
        nmi_max, stale, step = 0.0, 0, 0
        started = time.time()
        for epoch in range(config.epochs):
            for index in self.rng.permutation(len(self.train_batches)):
                terms = self._step(self.train_batches[index])
                if step % 20 == 0:
                    for key, value in terms.scalars().items():
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
            for k, value in enumerate(report["kl_w_per_dim"]):
                writer.add_scalar(f"val/kl_w_dim{k}", value, step)
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
                    seed=config.seed)
    return Trainer(config, data).fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    defaults = TrainConfig(dataset="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default=defaults.variant)
    parser.add_argument("--embeddings", default=defaults.embeddings)
    parser.add_argument("--run-name", default=None)
    for field in ("kappa", "omega", "alpha_z", "alpha_w", "alpha_a", "lr",
                  "val_fraction", "nmi_guard", "cov_ema", "grad_clip"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=float,
                            default=getattr(defaults, field))
    for field in ("d_z", "d_w", "hidden", "gat_dim", "heads",
                  "v_pcs", "epochs", "tile_cells", "patience", "eval_every",
                  "figures_every", "seed"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=int,
                            default=getattr(defaults, field))
    parser.add_argument("--phi-pca", type=int, default=None)
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
